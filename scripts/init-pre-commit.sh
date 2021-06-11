PROJECT_DIR=$(cd $(dirname $0);cd ..; pwd)
python3 -m venv ${PROJECT_DIR}/.venv_precommit  # tox と pre-commit を使うためだけの一時的な環境を作る
source ${PROJECT_DIR}/.venv_precommit/bin/activate
pip install pre-commit tox
tox -c ${PROJECT_DIR}/tox.ini -e lint
pre-commit install -c ${PROJECT_DIR}/scripts/.pre-commit-config.yaml
