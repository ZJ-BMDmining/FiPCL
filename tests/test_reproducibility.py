"""Regression checks for the release's actual input/sharing failure modes."""
from pathlib import Path
import sys

import torch
from torch import nn
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'FiPCL'), str(ROOT)]
import data
from models.fipcl_components import FisherInformationTracker, PrototypeBank
from models.shared_fusion_model import MRI_SHARING_GROUPS, create_independent_fipcl_model
from reproducibility import make_optimizer, subject_key


def test_linked_fold_cohort_resolution(tmp_path, monkeypatch):
    csv = tmp_path / 'MRI-Clinic-ADNI/fold_2_linked/train.csv'
    assert data._detect_mri_folder(csv).name == 'ADNI'
    cache = tmp_path / 'custom_cache'
    (cache / 'ADNI').mkdir(parents=True)
    monkeypatch.setattr(data, 'MRI_CACHE_DIR', cache)
    image = np.ones((39, 1, 8, 8), dtype=np.float32)
    np.save(cache / 'ADNI/subject.npy', image)
    frame = pd.DataFrame({'filename': ['subject.nii.gz']})
    store = data.CachedMRI(frame, data._detect_mri_folder(csv), target_h=8, target_w=8)
    assert torch.equal(store[0], torch.from_numpy(image))
    with pytest.raises(FileNotFoundError):
        data.CachedMRI(pd.DataFrame({'filename': ['absent.nii']}), 'ADNI')
    np.save(cache / 'ADNI/subject.npy', np.zeros_like(image))
    with pytest.raises(ValueError, match='all-zero'):
        store[0]
    image[0, 0, 0, 0] = np.nan
    np.save(cache / 'ADNI/subject.npy', image)
    with pytest.raises(ValueError, match='Nonfinite'):
        store[0]


class TinyMRI(nn.Module):
    """Only replaces the heavy image backbone in structural unit tests."""
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(4, 512)

    def forward(self, slices, **kwargs):
        features = self.projection(slices)
        return features[:, :3], features


@pytest.mark.parametrize('mri_mode', list(MRI_SHARING_GROUPS))
@pytest.mark.parametrize('tabular_mode', ['shared', 'independent'])
def test_sharing_and_optimizer_cross_task_updates(mri_mode, tabular_mode, monkeypatch, tmp_path):
    import model.slice_attention_resnet
    monkeypatch.setattr(model.slice_attention_resnet, 'Slice3DResNet', TinyMRI)
    torch.manual_seed(3)
    dims = {'clinic': 5, 'blood': 7, 'prot': 9}
    config = dict(input_dims=dims, embed_dim=16, dropout=0., device='cpu',
                  pretrained_mri=False, mri_sharing=mri_mode, tabular_sharing=tabular_mode,
                  clinic_encoder_type=None)
    model = create_independent_fipcl_model(**config)
    def encoder(mk):
        return model.mri_encoder if model.share_mri_encoder else model.mri_encoders[mk]
    for a in dims:
        for b in dims:
            assert (encoder(a) is encoder(b)) == (MRI_SHARING_GROUPS[mri_mode][a] == MRI_SHARING_GROUPS[mri_mode][b])
    assert (model.clinic_encoder.mlp is model.blood_encoder.mlp) == (tabular_mode == 'shared')
    assert model.clinic_encoder.input_proj is not model.blood_encoder.input_proj
    banks = [model.mri_prototype_bank(mk) for mk in dims]
    assert len({id(bank) for bank in banks}) == 3
    banks[0].update(torch.randn(3, 16), torch.tensor([0, 1, 2]))
    assert banks[0].initialized_flags.all()
    assert not banks[1].initialized_flags.any()
    optimizer = make_optimizer(model, learning_rate=.01)
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    for mk in dims:
        optimizer.zero_grad(set_to_none=True)
        batch = {'mri_slices': torch.randn(3, 4), mk: torch.randn(3, dims[mk])}
        before = encoder(mk).projection.weight.detach().clone()
        outputs = model(batch, return_all=True)
        loss = nn.functional.cross_entropy(outputs['fusion_logits'], torch.tensor([0, 1, 2]))
        loss.backward()
        assert encoder(mk).projection.weight.grad.abs().sum() > 0
        assert getattr(model, f'{mk}_encoder').mlp[0].weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, encoder(mk).projection.weight)
    model.eval()
    expected = model(batch)['fusion_logits'].detach()
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    restored = create_independent_fipcl_model(**config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(restored(batch)['fusion_logits'], expected, rtol=0, atol=0)


def test_legacy_blood_prot_alias_builds_without_forward_reference(monkeypatch):
    import model.slice_attention_resnet
    monkeypatch.setattr(model.slice_attention_resnet, 'Slice3DResNet', TinyMRI)
    model = create_independent_fipcl_model(
        {'clinic': 5, 'blood': 7, 'prot': 9}, embed_dim=16, device='cpu',
        pretrained_mri=False, share_mri_encoder=False, blood_share_mri_with='prot')
    assert model.mri_encoders['blood'] is model.mri_encoders['prot']
    assert model.mri_encoders['clinic'] is not model.mri_encoders['blood']


def test_fisher_logging_does_not_advance_ema():
    tracker = FisherInformationTracker(num_modalities=2, device='cpu', ema_alpha=.5)
    tracker.accumulate([1., 2.])
    first = list(tracker.get_trace())
    for _ in range(10):
        assert list(tracker.get_trace()) == first
    tracker.reset()
    tracker.accumulate([3., 4.])
    expected = .5 * np.array(first) + .5 * np.log1p([3., 4.])
    np.testing.assert_allclose(tracker.get_trace(), expected)
    np.testing.assert_allclose(tracker.get_trace(), expected)


def test_prototype_contrastive_uses_cosine_not_magnitude():
    bank = PrototypeBank(embed_dim=8)
    vectors = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 0])
    loss = bank.compute_prototype_loss(vectors, labels)
    torch.testing.assert_close(bank.compute_prototype_loss(vectors * 100, labels), loss)


def test_prototype_initialization_accepts_identical_embeddings():
    bank = PrototypeBank(embed_dim=8)
    vectors = nn.functional.normalize(torch.ones(8, 8), dim=1)
    bank.update(vectors, torch.zeros(8, dtype=torch.long))
    assert bank.initialized_flags[0]
    assert torch.isfinite(bank.prototypes).all()
    torch.testing.assert_close(bank.prototypes[0], vectors[:4])


def test_prediction_metrics_have_independent_confusion_check():
    from scripts.eval_from_checkpoints import confusion_metrics
    from trainer import compute_metrics
    labels = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(3)[[0, 1, 1, 2, 2, 0]] * .8 + .2 / 3
    checked, matrix = confusion_metrics(labels, probs)
    assert matrix == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert checked == {'accuracy': .5, 'f1': .5, 'recall': .5, 'precision': .5}
    for key, value in checked.items():
        assert compute_metrics(labels, probs)[key] == value
    with pytest.raises(ValueError, match='Invalid class probabilities'):
        confusion_metrics(labels, probs * 2)


def test_validation_seal_is_stable_and_rejects_smoke(tmp_path):
    import json
    from scripts.run_5fold import seal_validation
    from reproducibility import write_json
    run = tmp_path / 'runs/fold_0'
    record = {'status': 'COMPLETE', 'diagnostic_only': False, 'test_evaluated': False,
              'artifact_sha256': {}, 'checkpoint_sha256': {},
              'best_validation': {mk: {'f1': .4, 'epoch': 6} for mk in ('clinic', 'blood', 'prot')}}
    write_json(run / 'result.json', record)
    write_json(run / 'config.json', {'warmup': 5})
    first = seal_validation(tmp_path, [0])
    assert seal_validation(tmp_path, [0]) == first
    record['diagnostic_only'] = True
    write_json(run / 'result.json', record)
    with pytest.raises(ValueError, match='Invalid validation-only'):
        seal_validation(tmp_path, [0])
    assert json.loads((tmp_path / 'selection.json').read_text()) == first


@pytest.mark.parametrize('clinic_encoder_type,tabular_sharing', [(None, 'shared'), ('mlp', 'independent')])
def test_sealed_evaluation_restores_and_exports_predictions(tmp_path, monkeypatch, clinic_encoder_type, tabular_sharing):
    import model.slice_attention_resnet
    from scripts import eval_from_checkpoints as evaluation
    from scripts.run_5fold import aggregate_test, seal_validation
    from trainer import save_state
    from reproducibility import sha256, write_json
    monkeypatch.setattr(model.slice_attention_resnet, 'Slice3DResNet', TinyMRI)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    run = tmp_path / 'runs/fold_0'
    run.mkdir(parents=True)
    splits = tmp_path / 'protected'
    original = tmp_path / 'original'
    dims = {'clinic': 5, 'blood': 7, 'prot': 9}
    mconfig = dict(input_dims=dims, embed_dim=16, dropout=0., num_classes=3,
                   pretrained_mri=False, mri_sharing='clinic_blood', tabular_sharing=tabular_sharing,
                   clinic_encoder_type=clinic_encoder_type)
    model = create_independent_fipcl_model(**mconfig, device='cpu')
    csv = splits / 'MRI-Clinic-Demo/test.csv'
    csv.parent.mkdir(parents=True)
    csv.write_text('filename,label\na.nii,0\n')
    write_json(splits / 'data_manifest.json', {'source': str(original),
               'source_hashes': {str(original / 'MRI-Clinic-Demo/test.csv'): sha256(csv)}})
    write_json(run / 'config.json', {'seed': 42, 'threads': 1, 'amp': False,
                                    'split_root': str(splits), 'warmup': 1, 'fold': 0})
    write_json(run / 'manifest.json', {'data_manifest_sha256': sha256(splits / 'data_manifest.json')})
    torch.save({}, run / 'preprocessing.pt')
    checkpoints = {}
    for mk in dims:
        path = run / f'best_{mk}.pt'
        save_state(path, model, mconfig, 2, {})
        checkpoints[path.name] = sha256(path)
    write_json(run / 'result.json', {'status': 'COMPLETE', 'test_evaluated': False,
               'best_validation': {mk: {'f1': .4, 'epoch': 2} for mk in dims},
               'diagnostic_only': False, 'checkpoint_sha256': checkpoints,
               'artifact_sha256': {name: sha256(run / name) for name in ('config.json', 'manifest.json', 'preprocessing.pt')}})
    seal = tmp_path / 'selection.json'
    seal_validation(tmp_path, [0])
    def loader(args, mk, preprocessing, dimension):
        batch = {'mri_slices': torch.randn(3, 4), mk: torch.randn(3, dimension),
                 'label': torch.tensor([0, 1, 2])}
        return [batch], {str(csv): sha256(csv)}
    monkeypatch.setattr(evaluation, 'build_test_loader', loader)
    directory = tmp_path / 'evaluation/fold_0'
    report = evaluation.evaluate(run, directory, seal, workers=0)
    assert set(report['per_modality']) == set(dims)
    for mk in dims:
        with np.load(directory / f'test_{mk}.npz') as predictions:
            assert predictions['probabilities'].shape == (3, 3)
            verified, _ = evaluation.confusion_metrics(predictions['labels'], predictions['probabilities'])
        for key, value in verified.items():
            assert report['per_modality'][mk][key] == pytest.approx(value)
    summary = aggregate_test(tmp_path, [0])
    assert summary['folds'] == [0] and summary['selection_uses_test'] is False
    for mk in dims:
        for key, value in report['per_modality'][mk].items():
            assert summary['per_modality'][mk][key]['mean'] == value
            assert summary['per_modality'][mk][key]['std'] == 0.
    write_json(run / 'config.json', {'changed': True})
    with pytest.raises(ValueError, match='artifact changed'):
        evaluation.evaluate(run, tmp_path / 'rejected', seal, workers=0)


def test_proteomics_union_preserves_column_slots(tmp_path, monkeypatch):
    csv = tmp_path / 'MRI-Proteomics-ADNI/fold_0/train.csv'
    csv.parent.mkdir(parents=True)
    pd.DataFrame({'filename': ['a.nii', 'b.nii'], 'protein_b': [2., 6.], 'label': [0, 1]}).to_csv(csv, index=False)
    monkeypatch.setattr(data, '_preload_mri_data', lambda *a, **kw: [torch.ones(1)] * 2)
    ds = data.ProteomicsPairedDataset(csv, proteomics_cols=['protein_a', 'protein_b'], pad_to_dim=2)
    assert ds.proteomics_cols == ['protein_a', 'protein_b']
    np.testing.assert_array_equal(ds.feature_mask, [0., 1.])
    np.testing.assert_array_equal(ds.proteomics_data[:, 0], [0., 0.])
    single = data.SingleModalityProteomicsDataset(csv, scaler=ds.scaler,
                                                 proteomics_cols=ds.proteomics_cols, pad_to_dim=2)
    np.testing.assert_array_equal(single.feature_mask, ds.feature_mask)
    np.testing.assert_array_equal(single.prot_data, ds.proteomics_data)


def test_subject_identity_is_longitudinal_and_cross_task():
    assert subject_key('ADNI_009_S_1030_MR_scan1.nii', 'ADNI') == subject_key('ADNI_009_S_1030_MR_scan2.nii', 'ADNI')
    assert subject_key('PRGADC026_m00.nii', 'ANMerge') == subject_key('PRGADC026_m12.nii', 'ANMerge')
    assert subject_key('sub-OAS30734_ses-d0097_T1w.nii', 'OASIS') == subject_key('sub-OAS30734_ses-d0200_T1w.nii', 'OASIS')
    with pytest.raises(ValueError):
        subject_key('anonymous_uuid.nii', 'ANMerge')


@pytest.mark.parametrize('indicators', [False, True])
def test_clinic_union_roundtrip_uses_train_columns_scalers_and_slots(tmp_path, monkeypatch, indicators):
    from types import SimpleNamespace
    import experiment_data
    monkeypatch.setattr(data, '_preload_mri_data', lambda df, *a, **kw: [torch.ones(1)] * len(df))
    monkeypatch.setattr(experiment_data, 'PAIRS', {
        'clinic': ('Clinic', ('ADNI', 'AIBL'), data.ClinicPairedDataset,
                   data.scan_clinic_features, data.collate_clinic, 'clinic_cols')})
    frames = {
        'ADNI': pd.DataFrame({'filename': ['a.nii', 'b.nii'], 'Tesla': [1., 3.],
                              'mmse': [10., 30.], 'constant': [1., 1.], 'label': [0, 1]}),
        'AIBL': pd.DataFrame({'filename': ['c.nii', 'd.nii'], 'mmse': [20., 40.],
                              'age': [60., 80.], 'tesla': [1., 3.], 'label': [1, 2]}),
    }
    for cohort, frame in frames.items():
        folder = tmp_path / f'MRI-Clinic-{cohort}'
        (folder / 'fold_0_linked').mkdir(parents=True)
        frame.to_csv(folder / 'fold_0_linked/train.csv', index=False)
        # Extra validation-only columns must never enter the train-defined union.
        val = frame.assign(only_val=[100., 200.])
        val.loc[:, 'mmse'] += 10.
        val = val[list(reversed(val.columns))]
        val.to_csv(folder / 'fold_0_linked/val.csv', index=False)
        val.to_csv(folder / 'test.csv', index=False)
    args = SimpleNamespace(split_root=tmp_path, fold=0, clinic_feature_layout='union',
                           clinic_missing_indicator=indicators, no_pseudo=True,
                           batch_size=8, aux_batch_size=8, workers=0, seed=42)
    paired, _, dims, artifacts, _ = experiment_data.build_training_data(args)
    assert dims['clinic'] == 4 * (2 if indicators else 1)
    training = paired['clinic']['train'].dataset.datasets
    np.testing.assert_array_equal(training[0].clinic_data[:, :4], [[0, 0, -1, -1], [0, 0, 1, 1]])
    np.testing.assert_array_equal(training[1].clinic_data[:, :4], [[-1, 0, -1, -1], [1, 0, 1, 1]])
    saved = artifacts['clinic']['ADNI']
    assert saved['clinic_feature_union'] == ['age', 'constant', 'mmse', 'tesla']
    np.testing.assert_array_equal(saved['low_var_mask'], [True, True, False])
    if indicators:
        np.testing.assert_array_equal(training[0].clinic_data[:, 4:], [[0, 0, 1, 1]] * 2)
        np.testing.assert_array_equal(training[1].clinic_data[:, 4:], [[1, 0, 1, 1]] * 2)
    path = tmp_path / 'preprocessing.pt'
    torch.save(artifacts, path)
    restored = torch.load(path, weights_only=False)
    # Evaluation must use the serialized layout even if CLI defaults differ.
    args.clinic_feature_layout, args.clinic_missing_indicator = 'positional', False
    loader, _ = experiment_data.build_test_loader(args, 'clinic', restored, dims['clinic'])
    val = paired['clinic']['val'].dataset.datasets
    for original, loaded in zip(val, loader.dataset.datasets):
        np.testing.assert_array_equal(original.clinic_data, loaded.clinic_data)
    np.testing.assert_array_equal(loader.dataset.datasets[0].clinic_data[:, 2], [0, 2])
    np.testing.assert_array_equal(restored['clinic']['ADNI']['scaler'].mean_, saved['scaler'].mean_)


def test_clinic_layout_rejects_missing_or_ambiguous_saved_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(data, '_preload_mri_data', lambda df, *a, **kw: [torch.ones(1)] * len(df))
    csv = tmp_path / 'MRI-Clinic-ADNI/train.csv'
    csv.parent.mkdir()
    pd.DataFrame({'filename': ['a.nii', 'b.nii'], 'Tesla': [1., 3.],
                  'tesla': [1., 3.], 'label': [0, 1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='Ambiguous'):
        data.ClinicPairedDataset(csv)
    with pytest.raises(ValueError, match='Missing saved'):
        data.ClinicPairedDataset(csv, clinic_cols=['missing'])
    with pytest.raises(ValueError, match='require a semantic'):
        data.ClinicPairedDataset(csv, clinic_cols=['Tesla'], clinic_missing_indicator=True)
    with pytest.raises(ValueError, match='saved training scaler'):
        data.ClinicPairedDataset(csv, mode='test', clinic_cols=['Tesla'], clinic_feature_union=['tesla'])
    with pytest.raises(ValueError, match='unique casefolded'):
        data.ClinicPairedDataset(csv, clinic_cols=['Tesla'], clinic_feature_union=['Tesla'])


def test_clinic_mlp_has_gradients_and_rejects_shared_override(monkeypatch):
    import model.slice_attention_resnet
    from experiment_data import checkpoint_config
    from trainer import parse_args
    monkeypatch.setattr(model.slice_attention_resnet, 'Slice3DResNet', TinyMRI)
    dims = dict(clinic=5, blood=7, prot=9)
    args = parse_args(['--fold', '0', '--embed-dim', '16'])
    config = checkpoint_config(args, dims)
    assert config['clinic_encoder_type'] == 'mlp'
    torch.manual_seed(42)
    instance = create_independent_fipcl_model(**config, device='cpu')
    torch.manual_seed(42)
    default_instance = create_independent_fipcl_model(dims, embed_dim=16, device='cpu')
    assert default_instance.mri_sharing == default_instance.tabular_sharing == 'independent'
    assert default_instance.clinic_encoder_type == 'mlp'
    for key, value in instance.state_dict().items():
        torch.testing.assert_close(default_instance.state_dict()[key], value, rtol=0, atol=0)
    outputs = instance({'mri_slices': torch.randn(3, 4), 'clinic': torch.randn(3, 5)}, return_all=True)
    outputs['fusion_logits'].sum().backward()
    assert instance.clinic_encoder.encoder[0].weight.grad.abs().sum() > 0
    assert instance.mri_encoders['clinic'].projection.weight.grad.abs().sum() > 0
    assert instance.blood_encoder.mlp is not instance.prot_encoder.mlp
    with pytest.raises(ValueError, match='requires independent'):
        create_independent_fipcl_model(dims, clinic_encoder_type='mlp', tabular_sharing='shared')
    with pytest.raises(SystemExit):
        parse_args(['--fold', '0', '--clinic-encoder-type', 'mlp', '--tabular-sharing', 'shared'])


def test_fipcl_ramp_boundaries_and_default_coefficients():
    from trainer import parse_args, stage2_weight
    args = parse_args(['--fold', '0'])
    assert args.mri_sharing == args.tabular_sharing == 'independent'
    assert args.clinic_encoder_type == 'mlp'
    assert args.clinic_feature_layout == 'union' and args.clinic_missing_indicator
    assert args.early_stopping_task == 'clinic'
    assert args.warmup == 5
    assert stage2_weight(args, 5, 'clinic') == 0
    assert stage2_weight(args, 6, 'clinic') == pytest.approx(.01)
    assert stage2_weight(args, 15, 'clinic') == pytest.approx(.1)
    assert stage2_weight(args, 80, 'clinic') == pytest.approx(.1)
    assert stage2_weight(args, 6, 'blood') == pytest.approx(.1)
    assert stage2_weight(args, 15, 'prot') == 1
    args.stage2_ramp_epochs, args.clinic_stage2_scale = 0, 1.
    for task in ('clinic', 'blood', 'prot'):
        assert stage2_weight(args, 6, task) == 1
        assert stage2_weight(args, 90, task) == 1
    for bad in [['--clinic-stage2-scale', 'nan'], ['--stage2-ramp-epochs', '-1'],
                ['--clinic-stage2-scale', '0'], ['--clinic-missing-indicator']]:
        with pytest.raises(SystemExit):
            parse_args(['--fold', '0'] + bad)


def test_fixed_recipe_seal_rejects_mixed_configs_and_warmup(tmp_path):
    import json
    from reproducibility import write_json
    from scripts.run_5fold import seal_validation
    folds = list(range(5))
    for fold in folds:
        run = tmp_path / 'runs' / f'fold_{fold}'
        record = dict(status='COMPLETE', diagnostic_only=False, test_evaluated=False,
                      artifact_sha256={}, checkpoint_sha256={},
                      best_validation={task: dict(f1=.8, epoch=6) for task in ('clinic', 'blood', 'prot')})
        write_json(run / 'result.json', record)
        write_json(run / 'config.json', dict(warmup=5, seed=42, fold=fold))
    first = seal_validation(tmp_path, folds)
    assert first['validation_macro_f1']['clinic'] == .8
    assert seal_validation(tmp_path, folds) == first
    config = tmp_path / 'runs/fold_0/config.json'
    write_json(config, dict(warmup=5, seed=43, fold=0))
    with pytest.raises(ValueError, match='same training configuration'):
        seal_validation(tmp_path, folds)
    write_json(config, dict(warmup=5, seed=42, fold=0))
    path = tmp_path / 'runs/fold_0/result.json'
    changed = json.loads(path.read_text())
    changed['best_validation']['clinic']['epoch'] = 5
    write_json(path, changed)
    with pytest.raises(ValueError, match='Warmup-only'):
        seal_validation(tmp_path, folds)
