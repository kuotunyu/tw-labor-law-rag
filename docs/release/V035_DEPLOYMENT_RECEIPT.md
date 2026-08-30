# v0.3.5 Private Deployment Receipt

This receipt binds the reviewed application candidate to the observed private free-Space revision. The rotated runtime credential is intentionally non-exportable: candidate reads were observed through the running Space and collection scope was confirmed in the control plane, while no new local live denial probe was performed.

<!-- receipt-json:start -->
{
  "byok_fallback_policy": true,
  "candidate_source_sha": "7f38d6ec0fe4ba203dc0c7a2feadc691b4a02ae9",
  "collection_base": "labor_laws_20260830_3ec5ade",
  "date": "2026-08-30",
  "hardware": "cpu-basic",
  "no_provider_acceptance": {
    "api_contract_tests": true,
    "private_app_loaded": true,
    "provider_requests": 0,
    "remote_inventory_exact": true
  },
  "persistent_storage": false,
  "point_counts": {
    "fixed": 481,
    "structure": 884
  },
  "qdrant_runtime_evidence": {
    "candidate_read_observed": true,
    "legacy_scope_configured": true,
    "local_live_probe": false,
    "local_probe_reason": "runtime_credential_non_exportable_after_rotation"
  },
  "replicas": 1,
  "schema_version": "1.0",
  "space_policy_preflight": true,
  "space_revision_sha": "c441bd6e2d62705cb8cf7093e3de681320545fbc",
  "visibility": "private"
}
<!-- receipt-json:end -->
