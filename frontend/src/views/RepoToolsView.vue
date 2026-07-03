<script setup>
import { appState, applyFileEdit, cloneRepo, proposeFileEdit, readLocalFile, refreshRepos, resetSandboxApprovals } from "@/stores/appStore";
</script>

<template>
  <section class="page-fill page-single">
    <article class="card full-card">
      <header class="card-header">
        <div>
          <div class="eyebrow">Repository & File Tools</div>
          <h2 class="card-title">仓库与本地文件工具</h2>
          <div class="card-subtitle">克隆项目、查看本地仓库，并用模型生成受控文件修改提案。</div>
        </div>
        <el-button type="primary" plain @click="refreshRepos">刷新仓库</el-button>
      </header>

      <div class="card-body page-scroll">
        <div class="info-stat-grid">
          <div class="info-stat">
            <div class="label">本地仓库数量</div>
            <div class="value">{{ appState.repos.length }}</div>
          </div>
          <div class="info-stat">
            <div class="label">默认写入根目录</div>
            <div class="value compact">{{ appState.fileEditProposal?.writeRoot || "trl" }}</div>
          </div>
          <div class="info-stat">
            <div class="label">本窗口已授权命令</div>
            <div class="value compact">{{ Object.keys(appState.sandboxApprovals).length }}</div>
          </div>
        </div>

        <div class="section-block">
          <h3 class="section-title">沙盒授权</h3>
          <div class="status-note">
            本页的本地读取、生成修改提案、应用修改和克隆仓库都会先请求一次本窗口授权；同一个窗口里同一种命令确认一次后会记住。
          </div>
          <div class="inline-form">
            <el-tag v-for="(_, scope) in appState.sandboxApprovals" :key="scope" type="success">{{ scope }}</el-tag>
            <el-button plain @click="resetSandboxApprovals">清空本窗口授权</el-button>
          </div>
        </div>

        <div class="section-block">
          <h3 class="section-title">GitHub 克隆</h3>
          <div class="form-grid">
            <el-input v-model="appState.cloneForm.repoUrl" placeholder="GitHub 仓库地址，例如 https://github.com/user/repo.git" />
            <div class="inline-form">
              <el-input v-model="appState.cloneForm.branch" placeholder="分支，可选" />
              <el-input v-model="appState.cloneForm.targetName" placeholder="本地目录名，可选" />
              <el-button type="primary" :loading="appState.cloneLoading" @click="cloneRepo">开始克隆</el-button>
            </div>
            <div class="status-note">{{ appState.cloneStatus || "仓库会克隆到 trl/repos，便于后续检索和分析。" }}</div>
          </div>
        </div>

        <div class="section-block">
          <h3 class="section-title">本地文件读取</h3>
          <div class="form-grid">
            <el-input v-model="appState.fileReadForm.path" placeholder="文件路径，例如 PROJECT_OVERVIEW.md" />
            <div class="inline-form">
              <el-input-number v-model="appState.fileReadForm.startLine" :min="1" :max="999999" />
              <el-input-number v-model="appState.fileReadForm.endLine" :min="1" :max="999999" />
              <el-input-number v-model="appState.fileReadForm.maxBytes" :min="1000" :max="200000" :step="1000" />
              <el-button type="primary" :loading="appState.fileReadLoading" @click="readLocalFile">读取文件</el-button>
            </div>
            <div class="status-note">{{ appState.fileReadStatus || "读取只允许发生在 workspace 内，二进制和敏感文件会被拦截。" }}</div>
          </div>
          <div v-if="appState.fileReadResult" class="analysis-card">
            <div class="analysis-title">读取结果</div>
            <div class="repo-path">{{ appState.fileReadResult.path }}</div>
            <el-input :model-value="appState.fileReadResult.content || appState.fileReadResult.error || ''" type="textarea" :rows="12" readonly />
          </div>
        </div>

        <div class="section-block">
          <h3 class="section-title">受控文件修改</h3>
          <div class="form-grid">
            <el-input v-model="appState.fileEditForm.path" placeholder="文件路径，例如 frontend/src/views/RepoToolsView.vue" />
            <el-input
              v-model="appState.fileEditForm.instruction"
              type="textarea"
              :rows="4"
              placeholder="说明你想怎么改。系统会先生成完整修改提案，不会直接写入。"
            />
            <div class="inline-form">
              <el-input-number v-model="appState.fileEditForm.maxChars" :min="2000" :max="80000" :step="2000" />
              <el-button type="primary" :loading="appState.fileEditLoading" @click="proposeFileEdit">生成修改提案</el-button>
              <el-button
                type="success"
                :disabled="!appState.fileEditProposal?.changed || appState.fileEditProposal?.applied"
                :loading="appState.fileEditApplying"
                @click="applyFileEdit"
              >
                确认应用
              </el-button>
            </div>
            <div class="status-note">{{ appState.fileEditStatus || "权限范围：只能改 LOCAL_FILE_WRITE_ROOT 内已存在的文本文件；应用前会备份原文件。" }}</div>
          </div>

          <div v-if="appState.fileEditProposal" class="analysis-card">
            <div class="analysis-title">修改提案</div>
            <div class="status-note">{{ appState.fileEditProposal.summary }}</div>
            <div class="repo-path">{{ appState.fileEditProposal.path }}</div>
            <el-input
              :model-value="appState.fileEditProposal.diff || 'No diff.'"
              type="textarea"
              :rows="14"
              readonly
            />
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
          <div v-else class="empty-state">还没有检测到本地仓库，可以先克隆一个项目。</div>
        </div>
      </div>
    </article>
  </section>
</template>
