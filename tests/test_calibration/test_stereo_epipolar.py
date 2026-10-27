"""
Unit & Integration tests per la calibrazione stereo e il grafo epipolare (Modulo 02, Fase 3).
Metodologia: Test-Driven Development (TDD).
Verifica:
1. Dataclass StereoExtrinsics con serializzazione JSON e inversione di posa rigida.
2. Calcolo della distanza epipolare simmetrica punto-linea.
3. Grafo Epipolare con verifica analitica del Loop Closure (3 telecamere).
4. StereoPairCalibrator su dati sintetici con vincolo CALIB_FIX_INTRINSIC.
5. Calibrazione e coerenza stereo su dati reali multi-camera di galleria.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import pytest

from seeurn.calibration.monocular import CameraIntrinsics
from seeurn.calibration.stereo import (
    StereoExtrinsics,
    EpipolarGraph,
    StereoPairCalibrator,
    compute_epipolar_distance,
)
from seeurn.data.catalog import CalibrationCatalog
from seeurn.data.preprocessor import ImagePreprocessor
from seeurn.calibration.target_detector import RobustCheckerboardDetector


class TestStereoExtrinsics:
    """Verifica la dataclass StereoExtrinsics, proprietà algebriche e serializzazione JSON."""

    @pytest.fixture
    def sample_extrinsics(self) -> StereoExtrinsics:
        # Rotazione semplice di ~10 gradi attorno all'asse Y
        theta = np.radians(10.0)
        R = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ], dtype=np.float64)
        T = np.array([[-250.0], [15.0], [50.0]], dtype=np.float64)  # baseline in mm

        # Matrice antisimmetrica [T]x
        Tx = np.array([
            [0.0, -T[2, 0], T[1, 0]],
            [T[2, 0], 0.0, -T[0, 0]],
            [-T[1, 0], T[0, 0], 0.0],
        ], dtype=np.float64)

        # Matrice Essenziale E = [T]x @ R
        E = Tx @ R

        # K fittizia per calcolare F
        K = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        K_inv = np.linalg.inv(K)
        F = K_inv.T @ E @ K_inv

        return StereoExtrinsics(
            cam_i=1,
            cam_j=2,
            R=R,
            T=T,
            E=E,
            F=F,
            mean_epipolar_error_px=0.18,
        )

    def test_shapes_and_types(self, sample_extrinsics: StereoExtrinsics):
        assert sample_extrinsics.cam_i == 1
        assert sample_extrinsics.cam_j == 2
        assert sample_extrinsics.R.shape == (3, 3)
        assert sample_extrinsics.T.shape == (3, 1)
        assert sample_extrinsics.E.shape == (3, 3)
        assert sample_extrinsics.F.shape == (3, 3)
        assert isinstance(sample_extrinsics.mean_epipolar_error_px, float)

    def test_inverse_extrinsics(self, sample_extrinsics: StereoExtrinsics):
        inv = sample_extrinsics.invert()
        assert inv.cam_i == 2
        assert inv.cam_j == 1

        # R_inv = R^T
        np.testing.assert_allclose(inv.R, sample_extrinsics.R.T)
        # R @ R_inv = I
        np.testing.assert_allclose(sample_extrinsics.R @ inv.R, np.eye(3), atol=1e-7)

        # T_inv = -R^T @ T
        expected_T_inv = -sample_extrinsics.R.T @ sample_extrinsics.T
        np.testing.assert_allclose(inv.T, expected_T_inv)

        # F_inv = F^T
        np.testing.assert_allclose(inv.F, sample_extrinsics.F.T)

        # Doppia inversione deve ripristinare i valori originali
        double_inv = inv.invert()
        assert double_inv.cam_i == sample_extrinsics.cam_i
        assert double_inv.cam_j == sample_extrinsics.cam_j
        np.testing.assert_allclose(double_inv.R, sample_extrinsics.R)
        np.testing.assert_allclose(double_inv.T, sample_extrinsics.T)
        np.testing.assert_allclose(double_inv.F, sample_extrinsics.F)

    def test_json_serialization_roundtrip(self, sample_extrinsics: StereoExtrinsics, tmp_path: Path):
        file_path = tmp_path / "stereo_1_2_extrinsics.json"
        sample_extrinsics.save_to_json(file_path)
        assert file_path.exists()

        loaded = StereoExtrinsics.load_from_json(file_path)
        assert loaded.cam_i == sample_extrinsics.cam_i
        assert loaded.cam_j == sample_extrinsics.cam_j
        assert np.isclose(loaded.mean_epipolar_error_px, sample_extrinsics.mean_epipolar_error_px)
        np.testing.assert_allclose(loaded.R, sample_extrinsics.R)
        np.testing.assert_allclose(loaded.T, sample_extrinsics.T)
        np.testing.assert_allclose(loaded.E, sample_extrinsics.E)
        np.testing.assert_allclose(loaded.F, sample_extrinsics.F)


class TestEpipolarDistance:
    """Verifica il calcolo della distanza punto-linea epipolare simmetrica."""

    def test_exact_epipolar_constraint_zero_error(self):
        # Configurazione stereoscopica sintetica
        K1 = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        K2 = np.array([[1200.0, 0.0, 600.0], [0.0, 1200.0, 400.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        theta = np.radians(5.0)
        R = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ], dtype=np.float64)
        T = np.array([[-300.0], [10.0], [20.0]], dtype=np.float64)

        Tx = np.array([
            [0.0, -T[2, 0], T[1, 0]],
            [T[2, 0], 0.0, -T[0, 0]],
            [-T[1, 0], T[0, 0], 0.0],
        ], dtype=np.float64)
        E = Tx @ R
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

        # Generiamo punti 3D nello spazio della camera 1
        np.random.seed(42)
        P1 = np.random.uniform(low=[-200, -150, 800], high=[200, 150, 1500], size=(30, 3))
        
        # Proiezione ideale in Cam 1
        p1_h = (K1 @ P1.T).T
        p1 = p1_h[:, :2] / p1_h[:, 2:3]

        # Trasformazione e proiezione ideale in Cam 2: P2 = R @ P1 + T
        P2 = (R @ P1.T + T).T
        p2_h = (K2 @ P2.T).T
        p2 = p2_h[:, :2] / p2_h[:, 2:3]

        # La distanza epipolare simmetrica su proiezioni geometriche esatte deve essere numericamente ~0
        distances = compute_epipolar_distance(p1, p2, F)
        assert distances.shape == (30,)
        np.testing.assert_allclose(distances, 0.0, atol=1e-5)

    def test_perturbed_epipolar_distance(self):
        # Con una perturbazione nota di +2.0 px ortogonale, la distanza epipolare deve essere ~2.0 px
        K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        T = np.array([[-100.0], [0.0], [0.0]], dtype=np.float64)  # baseline puramente orizzontale
        Tx = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0], [0.0, -100.0, 0.0]], dtype=np.float64)
        E = Tx @ R
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

        p1 = np.array([[500.0, 500.0]], dtype=np.float64)
        # Linea epipolare in Cam 2 è perfettamente orizzontale a v = 500.0
        # Poniamo p2 a v = 503.0 (spostato di 3 pixel in verticale)
        p2 = np.array([[400.0, 503.0]], dtype=np.float64)

        d = compute_epipolar_distance(p1, p2, F)
        assert np.isclose(d[0], 3.0, atol=1e-3)


class TestEpipolarGraph:
    """Verifica il grafo epipolare e la chiusura del ciclo (Loop Closure)."""

    def test_loop_closure_exact_identity(self):
        graph = EpipolarGraph()

        # Definiamo trasformazioni coerenti a catena
        theta12 = np.radians(8.0)
        R12 = np.array([[np.cos(theta12), 0, np.sin(theta12)], [0, 1, 0], [-np.sin(theta12), 0, np.cos(theta12)]])
        T12 = np.array([[-200.0], [10.0], [30.0]])

        theta23 = np.radians(-15.0)
        R23 = np.array([[1, 0, 0], [0, np.cos(theta23), -np.sin(theta23)], [0, np.sin(theta23), np.cos(theta23)]])
        T23 = np.array([[150.0], [-100.0], [20.0]])

        # Trasformazione esatta chiusa 1 -> 3
        R13 = R23 @ R12
        T13 = R23 @ T12 + T23

        dummy_mat = np.eye(3)
        graph.add_edge(StereoExtrinsics(1, 2, R12, T12, dummy_mat, dummy_mat, 0.1))
        graph.add_edge(StereoExtrinsics(2, 3, R23, T23, dummy_mat, dummy_mat, 0.1))
        graph.add_edge(StereoExtrinsics(1, 3, R13, T13, dummy_mat, dummy_mat, 0.1))

        angle_deg, trans_mm = graph.check_loop_closure(c1=1, c2=2, c3=3)
        assert np.isclose(angle_deg, 0.0, atol=1e-5)
        assert np.isclose(trans_mm, 0.0, atol=1e-5)

    def test_loop_closure_detects_inconsistency(self):
        graph = EpipolarGraph()
        R = np.eye(3)
        T = np.array([[100.0], [0.0], [0.0]])
        dummy_mat = np.eye(3)

        graph.add_edge(StereoExtrinsics(1, 2, R, T, dummy_mat, dummy_mat, 0.1))
        graph.add_edge(StereoExtrinsics(2, 3, R, T, dummy_mat, dummy_mat, 0.1))

        # Modifichiamo T13 deliberatamente (errore di 5.0 mm su X) e ruotiamo di 2 gradi
        theta_err = np.radians(2.0)
        R_err = np.array([[np.cos(theta_err), -np.sin(theta_err), 0], [np.sin(theta_err), np.cos(theta_err), 0], [0, 0, 1]])
        T_err = np.array([[205.0], [0.0], [0.0]])

        graph.add_edge(StereoExtrinsics(1, 3, R_err, T_err, dummy_mat, dummy_mat, 0.1))

        angle_deg, trans_mm = graph.check_loop_closure(c1=1, c2=2, c3=3)
        assert np.isclose(angle_deg, 2.0, atol=1e-2)
        assert np.isclose(trans_mm, 5.0, atol=1e-2)

    def test_missing_edge_raises_value_error(self):
        graph = EpipolarGraph()
        dummy = np.eye(3)
        graph.add_edge(StereoExtrinsics(1, 2, dummy, np.zeros((3, 1)), dummy, dummy, 0.0))
        with pytest.raises(ValueError, match="Mancano archi"):
            graph.check_loop_closure(c1=1, c2=2, c3=3)


class TestStereoPairCalibratorSynthetic:
    """Verifica StereoPairCalibrator su scacchiera virtuale 3D sintetica."""

    @pytest.fixture
    def virtual_setup(self):
        K1 = np.array([[1500.0, 0.0, 1088.0], [0.0, 1500.0, 700.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        K2 = np.array([[1500.0, 0.0, 1088.0], [0.0, 1500.0, 700.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist1 = np.zeros(5, dtype=np.float64)
        dist2 = np.zeros(5, dtype=np.float64)

        intr1 = CameraIntrinsics(1, K1, dist1, (2176, 1400), 0.0)
        intr2 = CameraIntrinsics(2, K2, dist2, (2176, 1400), 0.0)

        # Trasformazione relativa nota: rotazione di 7 gradi e traslazione di 350 mm
        theta = np.radians(7.0)
        R_true = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ], dtype=np.float64)
        T_true = np.array([[-350.0], [20.0], [45.0]], dtype=np.float64)

        # Generiamo una griglia 15x9 (5mm)
        nx, ny = 15, 9
        objp = np.zeros((nx * ny, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * 5.0

        # Creiamo 15 pose casuali nello spazio di Cam 1
        np.random.seed(123)
        obj_list, img1_list, img2_list = [], [], []

        for _ in range(15):
            rvec1 = np.random.uniform(-0.15, 0.15, (3, 1))
            tvec1 = np.array([
                np.random.uniform(-100, 100),
                np.random.uniform(-80, 80),
                np.random.uniform(600, 900),
            ], dtype=np.float64).reshape(3, 1)

            # Proiezione in Cam 1
            pts1, _ = cv2.projectPoints(objp, rvec1, tvec1, K1, dist1)

            # Trasformazione in Cam 2: R2 = R_true @ R1, T2 = R_true @ T1 + T_true
            R1, _ = cv2.Rodrigues(rvec1)
            R2 = R_true @ R1
            T2 = R_true @ tvec1 + T_true
            rvec2, _ = cv2.Rodrigues(R2)

            # Proiezione in Cam 2
            pts2, _ = cv2.projectPoints(objp, rvec2, T2, K2, dist2)

            obj_list.append(objp.copy())
            img1_list.append(pts1.reshape(-1, 2).astype(np.float32))
            img2_list.append(pts2.reshape(-1, 2).astype(np.float32))

        return intr1, intr2, R_true, T_true, obj_list, img1_list, img2_list

    def test_calibrate_synthetic_recovers_ground_truth(self, virtual_setup):
        intr1, intr2, R_true, T_true, obj_list, img1_list, img2_list = virtual_setup
        calibrator = StereoPairCalibrator(intr1, intr2)

        extrinsics = calibrator.calibrate(
            object_points_list=obj_list,
            image_points_i_list=img1_list,
            image_points_j_list=img2_list,
            fix_intrinsics=True,
        )

        # Verifica rotazione
        R_diff = extrinsics.R @ R_true.T
        angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        assert angle_deg < 0.05, f"Errore di rotazione recovered: {angle_deg:.4f} gradi"

        # Verifica traslazione
        t_err = np.linalg.norm(extrinsics.T - T_true)
        assert t_err < 0.5, f"Errore di traslazione recovered: {t_err:.4f} mm"

        # Verifica errore epipolare sub-pixel su dati sintetici
        assert extrinsics.mean_epipolar_error_px < 0.05


class TestStereoCalibrationRealData:
    """
    Integrazione su dati reali di galleria del vento/cavitazione (wake2 dataset4).
    Calibra le 3 coppie stereoscopiche (1-2, 2-3, 1-3), verifica la chiusura del ciclo
    epipolare (Loop Closure) ed esporta tutti gli artefatti diagnostici.
    """

    RAW_ROOT = Path("data/raw/calibration")
    CFG_ROOT = Path("configs/calibration")
    OUT_ROOT = Path("tests/output/phase3_stereo")

    @pytest.mark.skipif(not RAW_ROOT.exists(), reason="Dati reali di calibrazione non presenti")
    def test_stereo_calibration_and_loop_closure(self):
        # 1. Carica le matrici intrinseche monoculari consolidate in Fase 2
        intr1 = CameraIntrinsics.load_from_json(self.CFG_ROOT / "camera_1_intrinsics.json")
        intr2 = CameraIntrinsics.load_from_json(self.CFG_ROOT / "camera_2_intrinsics.json")
        intr3 = CameraIntrinsics.load_from_json(self.CFG_ROOT / "camera_3_intrinsics.json")

        cat = CalibrationCatalog(self.RAW_ROOT)
        prep = ImagePreprocessor()
        detector = RobustCheckerboardDetector()

        triplets = cat.get_synchronized_triplets("wake2", "dataset4", require_all_views=True)
        assert len(triplets) > 100

        # Raccogliamo frame condivisi con visuale completa su tutte e 3 le telecamere
        obj_all, img1_all, img2_all, img3_all = [], [], [], []
        test_images_triplet = None
        test_points_triplet = None

        candidate_indices = [0, 10, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 400, 420, 440, 460, 480]
        for idx in candidate_indices:
            t = triplets[idx]
            im1 = prep.load_image(t.cam_paths[1], 1)
            im2 = prep.load_image(t.cam_paths[2], 2)
            im3 = prep.load_image(t.cam_paths[3], 3)
            d1 = detector.detect(im1, 1)
            d2 = detector.detect(im2, 2)
            d3 = detector.detect(im3, 3)
            if d1.is_valid and d2.is_valid and d3.is_valid:
                obj_all.append(d1.object_points)
                img1_all.append(d1.image_points)
                img2_all.append(d2.image_points)
                img3_all.append(d3.image_points)
                if test_images_triplet is None:
                    test_images_triplet = (im1, im2, im3)
                    test_points_triplet = (d1.image_points, d2.image_points, d3.image_points)

        assert len(obj_all) >= 15, f"Rilevate solo {len(obj_all)} triplette complete"

        # 2. Calibrazione Coppia (1, 2)
        calib12 = StereoPairCalibrator(intr1, intr2)
        ext12 = calib12.calibrate(obj_all, img1_all, img2_all, fix_intrinsics=True)
        assert ext12.mean_epipolar_error_px < 1.0, f"Errore epipolare 1-2 troppo alto: {ext12.mean_epipolar_error_px:.4f} px"
        assert 700.0 < ext12.baseline_mm < 950.0, f"Baseline 1-2 anomala: {ext12.baseline_mm:.2f} mm"

        # 3. Calibrazione Coppia (2, 3)
        calib23 = StereoPairCalibrator(intr2, intr3)
        ext23 = calib23.calibrate(obj_all, img2_all, img3_all, fix_intrinsics=True)
        assert ext23.mean_epipolar_error_px < 2.0, f"Errore epipolare 2-3 troppo alto: {ext23.mean_epipolar_error_px:.4f} px"
        assert 1000.0 < ext23.baseline_mm < 1500.0, f"Baseline 2-3 anomala: {ext23.baseline_mm:.2f} mm"

        # 4. Calibrazione Coppia (1, 3)
        calib13 = StereoPairCalibrator(intr1, intr3)
        ext13 = calib13.calibrate(obj_all, img1_all, img3_all, fix_intrinsics=True)
        assert ext13.mean_epipolar_error_px < 2.0, f"Errore epipolare 1-3 troppo alto: {ext13.mean_epipolar_error_px:.4f} px"
        assert 900.0 < ext13.baseline_mm < 1400.0, f"Baseline 1-3 anomala: {ext13.baseline_mm:.2f} mm"

        # 5. Costruzione Grafo Epipolare e Verifica Chiusura del Ciclo (Loop Closure)
        graph = EpipolarGraph()
        graph.add_edge(ext12)
        graph.add_edge(ext23)
        graph.add_edge(ext13)

        angle_deg, trans_err_mm = graph.check_loop_closure(1, 2, 3)
        assert angle_deg < 0.5, f"Errore rotazione Loop Closure troppo alto: {angle_deg:.4f} gradi"
        assert trans_err_mm < 5.0, f"Errore traslazione Loop Closure troppo alto: {trans_err_mm:.4f} mm"

        # 6. Esportazione Artefatti Metrici e Diagnostici Visivi
        im1, im2, im3 = test_images_triplet
        pts1, pts2, pts3 = test_points_triplet

        art12 = calib12.export_artifacts(
            ext12, im1, im2, pts1, pts2,
            output_dir=self.OUT_ROOT,
            config_dir=self.CFG_ROOT,
        )
        art23 = calib23.export_artifacts(
            ext23, im2, im3, pts2, pts3,
            output_dir=self.OUT_ROOT,
            config_dir=self.CFG_ROOT,
        )
        art13 = calib13.export_artifacts(
            ext13, im1, im3, pts1, pts3,
            output_dir=self.OUT_ROOT,
            config_dir=self.CFG_ROOT,
        )

        for art in (art12, art23, art13):
            assert art["json"].exists()
            assert art["overlay"].exists()
            assert art["report"].exists()

        # Salvataggio report complessivo di Loop Closure
        loop_report_path = self.OUT_ROOT / "loop_closure_report.md"
        with open(loop_report_path, "w", encoding="utf-8") as f:
            f.write("# Report Chiusura del Grafo Epipolare (Loop Closure)\n\n")
            f.write("Verifica della coerenza geometrica a 3 telecamere: C1 -> C2 -> C3 -> C1\n\n")
            f.write("## Metriche di Chiusura\n")
            f.write(f"- **Discrepanza di Rotazione (SO3):** `{angle_deg:.4f}°` (Soglia tolleranza: < 0.50°)\n")
            f.write(f"- **Discrepanza di Traslazione (Euclidea):** `{trans_err_mm:.4f} mm` (Soglia tolleranza: < 5.0 mm)\n")
            f.write(f"- **Errore Relativo su Baseline (~1 m):** `{(trans_err_mm / 1000.0) * 100:.2f}%`\n\n")
            f.write("## Sintesi Coppie Stereoscopiche\n")
            f.write(f"- **Stereo 1-2:** Baseline = `{ext12.baseline_mm:.2f} mm` | Errore Epipolare = `{ext12.mean_epipolar_error_px:.4f} px`\n")
            f.write(f"- **Stereo 2-3:** Baseline = `{ext23.baseline_mm:.2f} mm` | Errore Epipolare = `{ext23.mean_epipolar_error_px:.4f} px`\n")
            f.write(f"- **Stereo 1-3:** Baseline = `{ext13.baseline_mm:.2f} mm` | Errore Epipolare = `{ext13.mean_epipolar_error_px:.4f} px`\n\n")
            f.write("## Conclusioni per il Bundle Adjustment Globale\n")
            f.write("Il sistema di telecamere presenta una chiusura geometrica rigida eccezionale. ")
            f.write("Le trasformazioni pairwise sono pronte per costituire l'inizializzazione canonica nel Bundle Adjustment Globale (Fase 4).\n")

        assert loop_report_path.exists()


