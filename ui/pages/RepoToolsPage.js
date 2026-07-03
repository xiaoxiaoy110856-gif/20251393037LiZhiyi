import { appState, cloneRepo, refreshRepos } from "../store.js";

export default {
  name: "RepoToolsPage",
  setup() {
    return { appState, cloneRepo, refreshRepos };
  },
  template: `
    <section class="page-fill page-single">
      <article class="card full-card">
        <header class="card-header">
          <div>
            <div class="eyebrow">Repository Tools</div>
            <h2 class="card-title">仓库工具</h2>
            <div class="card-subtitle">在这里克隆 GitHub 仓库，并管理本地已经拉下来的项目目录。</div>
          </div>
          <el-button type="primary" plain @click="refreshRepos">刷新仓库列表</el-button>
        </header>
        <div class="card-body page-scroll">
          <div class="info-stat-grid">
            <div class="info-stat">
              <div class="label">本地项目数量</div>
              <div class="value">{{ appState.repos.length }}</div>
            </div>
            <div class="info-stat">
              <div class="label">默认目录</div>
              <div class="value compact">trl/repos</div>
            </div>
          </div>

          <div class="section-block">
            <h3 class="section-title">GitHub 克隆</h3>
            <div class="form-grid">
              <el-input v-model="appState.cloneForm.repoUrl" placeholder="例如：https://github.com/user/repo.git" />
              <div class="inline-form">
                <el-input v-model="appState.cloneForm.branch" placeholder="分支名（可选）" />
                <el-input v-model="appState.cloneForm.targetName" placeholder="本地目录名（可选）" />
                <el-button type="primary" :loading="appState.cloneLoading" @click="cloneRepo">克隆</el-button>
              </div>
              <div class="status-note">{{ appState.cloneStatus || '把 GitHub 仓库地址粘进来，克隆后的项目会进入 trl/repos。' }}</div>
            </div>
          </div>

          <div class="section-block">
            <h3 class="section-title">本地仓库列表</h3>
            <div v-if="appState.repos.length" class="repo-list">
              <div v-for="repo in appState.repos" :key="repo.path" class="repo-card">
                <div class="repo-name">{{ repo.name }}</div>
                <div class="repo-path">{{ repo.path }}</div>
              </div>
            </div>
            <div v-else class="empty-state">当前还没有本地仓库。等你克隆第一个项目后，这里会变成项目管理面板。</div>
          </div>
        </div>
      </article>
    </section>
  `,
};
