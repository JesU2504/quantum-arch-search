import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
	"test_qas_env_stop_reward",
	str(ROOT / "tests" / "test_qas_env_stop_reward.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print('Running test_stop_success_reward...')
mod.test_stop_success_reward()
print('OK')
print('Running test_stop_failure_reward...')
mod.test_stop_failure_reward()
print('OK')
print('All stop-reward tests passed')
