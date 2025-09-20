export FORGE_PROJECT_ROOT="/data0/matthew/Games/Forge"
source "${FORGE_PROJECT_ROOT}/cardforge-scripts/forge_env.sh"

cd forge-helper

mvn versions:update-properties \
  -DincludeProperties=forge.version \
  -DallowSnapshots=true \
  -DgenerateBackupPoms=false \
  -Dmaven.repo.local="${FORGE_PROJECT_ROOT}/maven_cache"

mvn -U clean package \
  -Dmaven.repo.local="${FORGE_PROJECT_ROOT}/maven_cache"
