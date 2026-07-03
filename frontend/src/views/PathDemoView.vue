<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import L from "leaflet";
import { api } from "@/api";

// 核心6：真实道路 baseline。地图页先通过 OSRM 获取路线，再在同一路线上叠加 DQN/PPO 策略标记。
const OSRM_URL = "https://router.project-osrm.org/route/v1/driving";
const MAP_BOUNDS = [
  [30.548, 104.244],
  [30.6035, 104.2835],
];

const mapRoot = ref(null);
const mapReady = ref(false);
const selectingPoints = ref(false);
const selectedPoints = ref([]);
const routeReady = ref(false);
const saving = ref(false);
const savedTrajectoryId = ref(null);
const routeStatus = ref("地图已就绪。请先点击“开始标注”，再依次点击起点和终点。");
const saveStatus = ref("尚未生成轨迹。");

const methodCategory = ref("rl");
const selectedRlMethod = ref("baseline");
const selectedCompressionMethod = ref("s3");
const scenarioId = ref("campus");

const routeGeometry = ref([]);
const routeDistanceKm = ref(0);
const routeDurationMin = ref(0);
const routeSteps = ref([]);
const dqnDecisionPoints = ref([]);
const ppoStages = ref([]);
// 核心7：S3/RLTS/Mlsimp 展示状态。每个算法保存自己保留的路线点索引，地图据此高亮压缩结果。
const compressionStrategies = ref({
  s3: { indices: [], summary: "S3 倾向保留转向点和少量骨架点。" },
  rlts: { indices: [], summary: "RLTS 更关注转折和局部行为变化，通常在关键路口保留更多点。" },
  mlsimp: { indices: [], summary: "Mlsimp 会保留端点与长段变化点，整体点数更克制。" },
});
const playbackActive = ref(false);
const playbackIndex = ref(0);
const recentTrajectories = ref([]);

const presetScenarios = [
  {
    id: "campus",
    label: "师大双校区示例轨迹",
    start: [30.54908, 104.24808],
    end: [30.59883, 104.26942],
    description: "从狮子山校区北门方向到成龙校区南门方向的示例。",
  },
  {
    id: "eastlake",
    label: "东安湖片区轨迹",
    start: [30.59402, 104.24825],
    end: [30.59957, 104.26778],
    description: "东安湖一带到龙泉主城主干路的轨迹样例。",
  },
  {
    id: "industry",
    label: "经开区工业片区轨迹",
    start: [30.56966, 104.24543],
    end: [30.58954, 104.27291],
    description: "工业片区主干路间的样例轨迹。",
  },
];

const compressionOptions = [
  { label: "S3", value: "s3" },
  { label: "RLTS", value: "rlts" },
  { label: "Mlsimp", value: "mlsimp" },
];

const rlOptions = [
  { label: "Baseline", value: "baseline" },
  { label: "DQN", value: "dqn" },
  { label: "PPO", value: "ppo" },
];

const activeScenario = computed(
  () => presetScenarios.find((item) => item.id === scenarioId.value) ?? presetScenarios[0],
);

const effectiveEndpoints = computed(() => {
  // 根据手动选点或预设场景决定本次轨迹的起点、终点和标签。
  if (selectedPoints.value.length === 2) {
    return {
      type: "manual",
      scenarioId: "",
      scenarioLabel: "手动标注轨迹",
      start: selectedPoints.value[0].coords,
      end: selectedPoints.value[1].coords,
    };
  }
  return {
    type: "preset",
    scenarioId: activeScenario.value.id,
    scenarioLabel: activeScenario.value.label,
    start: activeScenario.value.start,
    end: activeScenario.value.end,
  };
});

const currentCompression = computed(
  // 当前选中的压缩算法结果，用于地图渲染和压缩率统计。
  () => compressionStrategies.value[selectedCompressionMethod.value] ?? { indices: [], summary: "" },
);

const currentSteps = computed(() => {
  // 右侧步骤说明：根据当前模式展示 baseline、DQN、PPO 或压缩算法解释。
  if (methodCategory.value === "compression") {
    return currentCompression.value.indices.map((routeIndex, index) => ({
      name: `保留点 ${index + 1}`,
      note: `该点对应原始轨迹第 ${routeIndex + 1} 个采样点，作为 ${selectedCompressionMethod.value.toUpperCase()} 的关键保留点。`,
    }));
  }
  if (selectedRlMethod.value === "baseline") {
    return [
      {
        name: "Baseline",
        note: "Baseline 只展示严格贴合路网的真实道路路径，作为后续 DQN 和 PPO 的共同参照。",
      },
    ];
  }
  if (selectedRlMethod.value === "dqn") {
    const visible = playbackActive.value ? playbackIndex.value : dqnDecisionPoints.value.length;
    return dqnDecisionPoints.value.slice(0, visible).map((item, index) => ({
      name: `DQN 决策点 ${index + 1}`,
      note: item.note,
    }));
  }
  const visible = playbackActive.value ? playbackIndex.value : ppoStages.value.length;
  return ppoStages.value.slice(0, visible).map((item, index) => ({
    name: `PPO 阶段 ${index + 1}`,
    note: item.note,
  }));
});

const trajectoryHint = computed(() => {
  if (!selectingPoints.value && selectedPoints.value.length === 0) {
    return "当前没有手动标注。你可以直接使用预存轨迹，或者点击“开始标注”选择起点和终点。";
  }
  if (selectedPoints.value.length === 0) return "请点击地图选择起点。";
  if (selectedPoints.value.length === 1) return "起点已选择，请继续点击终点。";
  return "起点和终点都已选好。点击“生成轨迹”会按真实路网生成 baseline，并保存到数据库。";
});

const compressionStats = computed(() => {
  // 计算当前压缩算法保留点数、总点数和压缩比例。
  if (methodCategory.value !== "compression" || !routeGeometry.value.length) return null;
  const kept = currentCompression.value.indices.length;
  const total = routeGeometry.value.length;
  return {
    kept,
    total,
    ratio: total ? `${((kept / total) * 100).toFixed(1)}%` : "0%",
  };
});

const canGenerateRoute = computed(() => selectedPoints.value.length === 2 || effectiveEndpoints.value.type === "preset");
const canReplay = computed(() => methodCategory.value === "rl" && routeReady.value && selectedRlMethod.value !== "baseline");

let map = null;
let tileLayer = null;
let routeLayer = null;
let endpointLayer = null;
let selectionLayer = null;
let playbackTimer = null;

function stopPlayback() {
  // 停止 DQN/PPO 播放动画，恢复完整展示。
  if (playbackTimer) clearInterval(playbackTimer);
  playbackTimer = null;
  playbackActive.value = false;
}

function clearRouteState() {
  // 清空当前路线、DQN/PPO 点、压缩结果和保存状态。
  stopPlayback();
  routeReady.value = false;
  routeGeometry.value = [];
  routeDistanceKm.value = 0;
  routeDurationMin.value = 0;
  routeSteps.value = [];
  dqnDecisionPoints.value = [];
  ppoStages.value = [];
  savedTrajectoryId.value = null;
  compressionStrategies.value = {
    s3: { indices: [], summary: "S3 倾向保留转向点和少量骨架点。" },
    rlts: { indices: [], summary: "RLTS 更关注转折和局部行为变化，通常在关键路口保留更多点。" },
    mlsimp: { indices: [], summary: "Mlsimp 会保留端点与长段变化点，整体点数更克制。" },
  };
}

function ensureLayer(currentLayer) {
  // 获取或重建 Leaflet 图层，避免重复绘制时旧图层残留。
  if (currentLayer && map) {
    map.removeLayer(currentLayer);
  }
  return map ? L.layerGroup().addTo(map) : null;
}

function resetSelectionForNewTrajectory(point) {
  // 重新开始手动标注时清空旧选点，并写入新的起点。
  selectedPoints.value = [point];
  clearRouteState();
  routeStatus.value = "已开始一条新的手动轨迹。请继续点击终点。";
  saveStatus.value = "等待生成轨迹。";
}

function renderSelectedPoints() {
  // 在地图上渲染用户手动选择的起点和终点。
  if (!map) return;
  selectionLayer = ensureLayer(selectionLayer);
  selectedPoints.value.forEach((point, index) => {
    const label = index === 0 ? "起点" : "终点";
    const color = index === 0 ? "#ff547a" : "#ff9b4c";
    L.circleMarker(point.coords, {
      radius: 7,
      color: "#ffffff",
      weight: 2,
      fillColor: color,
      fillOpacity: 1,
    })
      .bindTooltip(label, { permanent: true, direction: "top", className: "route-point-tooltip" })
      .addTo(selectionLayer);
  });
}

function renderRouteEndpoints() {
  // 在地图上渲染当前路线的起终点。
  if (!map) return;
  endpointLayer = ensureLayer(endpointLayer);
  if (!routeReady.value) return;
  const [startCoords, endCoords] = [effectiveEndpoints.value.start, effectiveEndpoints.value.end];
  [
    { label: "起点", coords: startCoords, color: "#ff547a" },
    { label: "终点", coords: endCoords, color: "#ff9b4c" },
  ].forEach((item) => {
    L.circleMarker(item.coords, {
      radius: 8,
      color: "#ffffff",
      weight: 2,
      fillColor: item.color,
      fillOpacity: 1,
    })
      .bindTooltip(item.label, { permanent: true, direction: "top", className: "route-point-tooltip" })
      .addTo(endpointLayer);
  });
}

function buildCompressionStrategies() {
  // 核心7：S3/RLTS/Mlsimp 的演示实现。结合转向点和均匀骨架点，生成可对比的保留点集合。
  const total = routeGeometry.value.length;
  if (!total) return;
  const turnIndices = dqnDecisionPoints.value.map((item) => item.routeIndex);
  const uniqueSorted = (values) => Array.from(new Set(values.filter((value) => value >= 0 && value < total))).sort((a, b) => a - b);

  const s3 = uniqueSorted([
    0,
    total - 1,
    ...turnIndices,
    ...Array.from({ length: 8 }, (_, idx) => Math.round((idx * (total - 1)) / 7)),
  ]);

  const rlts = uniqueSorted([
    0,
    total - 1,
    ...turnIndices.flatMap((idx) => [idx - 1, idx, idx + 1]),
    ...Array.from({ length: 10 }, (_, idx) => Math.round((idx * (total - 1)) / 9)),
  ]);

  const mlsimp = uniqueSorted([
    0,
    total - 1,
    ...turnIndices.filter((_, index) => index % 2 === 0),
    ...Array.from({ length: 6 }, (_, idx) => Math.round((idx * (total - 1)) / 5)),
  ]);

  compressionStrategies.value = {
    s3: { indices: s3, summary: "S3 保留端点、明显转折点和少量均匀骨架点。" },
    rlts: { indices: rlts, summary: "RLTS 更密集地保留路口附近的关键点，强调局部行为变化。" },
    mlsimp: { indices: mlsimp, summary: "Mlsimp 以端点和代表性长段变化点为主，整体保留点更精炼。" },
  };
}

function buildDqnDecisionPoints(steps) {
  // 核心6：DQN 展示。把 OSRM 返回的转向/路口信息转换成离散决策点。
  const total = routeGeometry.value.length || 1;
  return steps
    .filter((step) => step.maneuver?.location)
    .map((step, index) => {
      const coords = [step.maneuver.location[1], step.maneuver.location[0]];
      return {
        coords,
        routeIndex: Math.min(total - 1, Math.max(0, Math.round(((index + 1) / Math.max(steps.length, 1)) * (total - 1)))),
        note: `${step.name || "未命名道路"}，动作=${step.maneuver.type || "continue"}，距离约 ${(step.distance / 1000).toFixed(2)} km。`,
      };
    });
}

function buildPpoStages(steps) {
  // 核心6：PPO 展示。把整条路线按阶段分组，表现“阶段策略”而不是单个路口动作。
  if (!steps.length) return [];
  const chunkSize = Math.max(1, Math.ceil(steps.length / 4));
  const groups = [];
  for (let index = 0; index < steps.length; index += chunkSize) {
    groups.push(steps.slice(index, index + chunkSize));
  }
  return groups.map((group, index) => {
    const last = group[group.length - 1];
    const distance = group.reduce((sum, item) => sum + (item.distance || 0), 0);
    return {
      coords: [last.maneuver.location[1], last.maneuver.location[0]],
      note: `阶段 ${index + 1} 聚合 ${group.length} 个动作，主要沿 ${group[0].name || "当前主路"} 前进，距离约 ${(distance / 1000).toFixed(2)} km。`,
    };
  });
}

function renderBaseline(targetLayer, color = "#3b6cff") {
  // 绘制蓝色 baseline 路线，作为 DQN/PPO/压缩算法共同参照。
  if (!routeGeometry.value.length) return;
  L.polyline(routeGeometry.value, {
    color,
    weight: 6,
    opacity: 0.92,
  }).addTo(targetLayer);
}

function renderDqn(targetLayer) {
  // 核心6：把 DQN 画成红色离散决策链，并叠加在蓝色 baseline 上。
  renderBaseline(targetLayer, "#3b6cff");
  const visible = playbackActive.value ? playbackIndex.value : dqnDecisionPoints.value.length;
  const points = dqnDecisionPoints.value.slice(0, visible);
  if (!points.length) return;
  const line = [routeGeometry.value[0], ...points.map((item) => item.coords)];
  L.polyline(line, {
    color: "#ff547a",
    weight: 5,
    opacity: 0.96,
  }).addTo(targetLayer);
  points.forEach((item, index) => {
    L.circleMarker(item.coords, {
      radius: 6,
      color: "#ffffff",
      weight: 2,
      fillColor: "#ff547a",
      fillOpacity: 1,
    })
      .bindTooltip(`DQN-${index + 1}`, { permanent: true, direction: "top", className: "route-point-tooltip" })
      .addTo(targetLayer);
  });
}

function renderPpo(targetLayer) {
  // 核心6：把 PPO 画成绿色分阶段路径，用虚线表现阶段策略。
  renderBaseline(targetLayer, "#3b6cff");
  const visible = playbackActive.value ? playbackIndex.value : ppoStages.value.length;
  const stages = ppoStages.value.slice(0, visible);
  if (!stages.length) return;
  const line = [routeGeometry.value[0], ...stages.map((item) => item.coords)];
  L.polyline(line, {
    color: "#45c38a",
    weight: 5,
    opacity: 0.96,
    dashArray: "12 10",
  }).addTo(targetLayer);
  stages.forEach((item, index) => {
    L.circleMarker(item.coords, {
      radius: 7,
      color: "#ffffff",
      weight: 2,
      fillColor: "#45c38a",
      fillOpacity: 1,
    })
      .bindTooltip(`PPO-${index + 1}`, { permanent: true, direction: "top", className: "route-point-tooltip" })
      .addTo(targetLayer);
  });
}

function renderCompression(targetLayer) {
  // 核心7：先淡化显示原始路线点，再高亮当前 S3/RLTS/Mlsimp 保留下来的关键点。
  renderBaseline(targetLayer, "rgba(59,108,255,0.55)");
  routeGeometry.value.forEach((coords) => {
    L.circleMarker(coords, {
      radius: 2.2,
      color: "#91a4d6",
      weight: 0,
      fillColor: "#91a4d6",
      fillOpacity: 0.35,
    }).addTo(targetLayer);
  });

  const palette = { s3: "#875cff", rlts: "#ff8a4c", mlsimp: "#45c38a" };
  const color = palette[selectedCompressionMethod.value];
  const keptCoords = currentCompression.value.indices.map((index) => routeGeometry.value[index]).filter(Boolean);
  if (keptCoords.length > 1) {
    L.polyline(keptCoords, {
      color,
      weight: 3,
      opacity: 0.88,
      dashArray: "10 8",
    }).addTo(targetLayer);
  }
  keptCoords.forEach((coords, index) => {
    L.circleMarker(coords, {
      radius: 6,
      color: "#ffffff",
      weight: 2,
      fillColor: color,
      fillOpacity: 1,
    })
      .bindTooltip(`保留点 ${index + 1}`, { direction: "top", className: "route-point-tooltip" })
      .addTo(targetLayer);
  });
}

function renderRoute() {
  // 根据当前模式统一分发地图绘制：压缩算法、DQN、PPO 或 baseline。
  if (!map) return;
  routeLayer = ensureLayer(routeLayer);
  if (!routeReady.value || !routeGeometry.value.length) return;

  if (methodCategory.value === "compression") {
    renderCompression(routeLayer);
    return;
  }
  if (selectedRlMethod.value === "dqn") {
    renderDqn(routeLayer);
    return;
  }
  if (selectedRlMethod.value === "ppo") {
    renderPpo(routeLayer);
    return;
  }
  renderBaseline(routeLayer);
}

function startPlayback(mode = selectedRlMethod.value) {
  // 播放 DQN/PPO 策略过程，让地图按决策点或阶段逐步显示。
  stopPlayback();
  if (methodCategory.value !== "rl" || mode === "baseline" || !routeReady.value) {
    renderRoute();
    return;
  }
  const items = mode === "dqn" ? dqnDecisionPoints.value : ppoStages.value;
  if (!items.length) return;
  playbackActive.value = true;
  playbackIndex.value = 0;
  renderRoute();
  playbackTimer = setInterval(() => {
    playbackIndex.value += 1;
    renderRoute();
    if (playbackIndex.value >= items.length) stopPlayback();
  }, mode === "dqn" ? 700 : 950);
}

async function loadRecentTrajectories() {
  // 从后端读取最近保存的地图实验记录。
  try {
    const data = await api.trajectories(10);
    recentTrajectories.value = data.items || [];
  } catch {
    recentTrajectories.value = [];
  }
}

async function saveCurrentTrajectory() {
  // 核心6/7：保存当前地图实验，包括 RL 方法或压缩方法，方便后端/数据库展示最近记录。
  if (!routeGeometry.value.length) return;
  saving.value = true;
  try {
    const data = await api.saveTrajectory({
      trajectory_type: effectiveEndpoints.value.type,
      scenario_id: effectiveEndpoints.value.scenarioId,
      scenario_label: effectiveEndpoints.value.scenarioLabel,
      rl_method: methodCategory.value === "rl" ? selectedRlMethod.value : "",
      compression_method: methodCategory.value === "compression" ? selectedCompressionMethod.value : "",
      map_provider: "OpenStreetMap",
      route_provider: "OSRM",
      start: effectiveEndpoints.value.start,
      end: effectiveEndpoints.value.end,
      distance_km: routeDistanceKm.value,
      duration_min: routeDurationMin.value,
      route_geometry: routeGeometry.value,
      compression: compressionStrategies.value,
      metadata: {
        category: methodCategory.value,
        selected_points: selectedPoints.value.map((item) => item.coords),
      },
    });
    savedTrajectoryId.value = data.trajectoryId || null;
    saveStatus.value = data.trajectoryId ? `轨迹已保存到数据库，ID = ${data.trajectoryId}` : "轨迹已生成，但数据库当前未启用。";
    await loadRecentTrajectories();
  } catch (error) {
    saveStatus.value = `轨迹保存失败：${error.message}`;
  } finally {
    saving.value = false;
  }
}

async function fetchRoadRoute() {
  // 核心6：从 OSRM 获取贴合真实道路网络的 baseline 轨迹。
  const { start, end } = effectiveEndpoints.value;
  const startLngLat = `${start[1]},${start[0]}`;
  const endLngLat = `${end[1]},${end[0]}`;
  const url = `${OSRM_URL}/${startLngLat};${endLngLat}?overview=full&geometries=geojson&steps=true`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`OSRM 请求失败：${response.status}`);
  const payload = await response.json();
  const route = payload.routes?.[0];
  if (!route?.geometry?.coordinates?.length) {
    throw new Error("没有拿到有效的道路路径。");
  }
  return route;
}

async function generateTrajectory() {
  // 核心6/7：完整地图生成流程：OSRM baseline -> DQN/PPO 标记 -> S3/RLTS/Mlsimp 保留点 -> 可选保存数据库。
  stopPlayback();
  if (!canGenerateRoute.value) {
    routeStatus.value = "请先选择起点和终点，或者选择一条预存轨迹。";
    return;
  }
  routeStatus.value = "正在按照真实路网生成 baseline...";
  saveStatus.value = "生成成功后会自动写入 MySQL。";
  try {
    const route = await fetchRoadRoute();
    routeGeometry.value = route.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
    routeDistanceKm.value = Number((route.distance / 1000).toFixed(2));
    routeDurationMin.value = Number((route.duration / 60).toFixed(1));
    routeSteps.value = route.legs?.[0]?.steps || [];
    dqnDecisionPoints.value = buildDqnDecisionPoints(routeSteps.value);
    ppoStages.value = buildPpoStages(routeSteps.value);
    buildCompressionStrategies();
    routeReady.value = true;
    routeStatus.value = "真实路网 baseline 已生成。";
    renderRouteEndpoints();
    renderRoute();
    if (map && routeGeometry.value.length) {
      map.fitBounds(L.polyline(routeGeometry.value).getBounds(), { padding: [28, 28] });
    }
    await saveCurrentTrajectory();
    if (methodCategory.value === "rl" && selectedRlMethod.value !== "baseline") {
      startPlayback(selectedRlMethod.value);
    }
  } catch (error) {
    clearRouteState();
    routeStatus.value = `轨迹生成失败：${error.message}`;
    saveStatus.value = "未保存。";
    renderRouteEndpoints();
    renderRoute();
  }
}

function toggleSelectingPoints() {
  // 开关手动选点模式，用于用户自定义轨迹起点和终点。
  selectingPoints.value = !selectingPoints.value;
  routeStatus.value = selectingPoints.value
    ? "标注已开启。请先点击起点，再点击终点。"
    : "标注已关闭。";
}

function clearSelectedPoints() {
  // 清空手动选择的起终点，并恢复预设场景路线。
  selectingPoints.value = false;
  selectedPoints.value = [];
  clearRouteState();
  routeStatus.value = "已清空标注和轨迹。";
  saveStatus.value = "等待生成新的轨迹。";
  renderSelectedPoints();
  renderRouteEndpoints();
  renderRoute();
}

function handleMapClick(event) {
  // 地图点击事件：在手动选点模式下记录起点/终点坐标。
  if (!selectingPoints.value) return;
  const point = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    coords: [event.latlng.lat, event.latlng.lng],
  };
  if (selectedPoints.value.length >= 2) {
    resetSelectionForNewTrajectory(point);
  } else {
    selectedPoints.value = [...selectedPoints.value, point];
    routeStatus.value = selectedPoints.value.length === 1 ? "起点已选择，请继续点击终点。" : "起点和终点都已选择，现在可以生成轨迹。";
  }
  renderSelectedPoints();
  renderRouteEndpoints();
  renderRoute();
}

async function ensureMap() {
  // 初始化 Leaflet 地图、底图、图层和点击事件。
  if (!mapRoot.value) return;
  await nextTick();
  if (map) {
    map.off("click", handleMapClick);
    map.remove();
  }
  map = L.map(mapRoot.value, {
    zoomControl: true,
    preferCanvas: true,
  });
  tileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  map.fitBounds(MAP_BOUNDS, { padding: [20, 20] });
  map.on("click", handleMapClick);
  L.control.scale().addTo(map);
  mapReady.value = true;
  renderSelectedPoints();
  renderRouteEndpoints();
  renderRoute();
  setTimeout(() => map?.invalidateSize(true), 50);
}

watch(methodCategory, () => {
  stopPlayback();
  renderRoute();
});

watch(selectedRlMethod, (value) => {
  if (routeReady.value && value !== "baseline" && methodCategory.value === "rl") {
    startPlayback(value);
  } else {
    stopPlayback();
    renderRoute();
  }
});

watch(selectedCompressionMethod, () => {
  if (methodCategory.value === "compression") {
    renderRoute();
  }
});

watch(scenarioId, () => {
  if (selectedPoints.value.length === 2) return;
  clearRouteState();
  routeStatus.value = "预存轨迹已切换。点击“生成轨迹”后会严格按路网生成并保存。";
  saveStatus.value = "等待生成新的轨迹。";
  renderRoute();
  renderRouteEndpoints();
});

onMounted(async () => {
  await ensureMap();
  await loadRecentTrajectories();
});

onBeforeUnmount(() => {
  stopPlayback();
  if (map) {
    map.off("click", handleMapClick);
    map.remove();
  }
});
</script>

<template>
  <section class="page-fill page-single">
    <article class="card full-card">
      <header class="card-header">
        <div>
          <div class="eyebrow">RL Path Planning</div>
          <h2 class="card-title">路径规划与轨迹压缩演示</h2>
          <div class="card-subtitle">
            先在地图上选择起点和终点，再生成严格贴合路网的 baseline。随后可以用 DQN、PPO 或轨迹压缩算法查看不同策略。
          </div>
        </div>
      </header>

      <div class="card-body page-scroll">
        <div class="policy-explainer-grid path-top-notes">
          <div class="policy-note">
            <h3>地图上的展示逻辑</h3>
            <p>
              蓝色 baseline 来自真实道路网络。切到 DQN 时会按路口逐点展示决策；切到 PPO 时会按阶段展示策略；切到轨迹压缩时，会在原始轨迹上高亮不同算法保留的关键点。
            </p>
          </div>
          <div class="policy-note">
            <h3>当前要求</h3>
            <p>
              你选择起点和终点后，系统会按路网生成 baseline，并把生成结果和压缩策略一起保存到 MySQL，方便后续分析和复现。
            </p>
          </div>
        </div>

        <div class="control-grid">
          <div class="analysis-card">
            <div class="analysis-label">方法类别</div>
            <el-select v-model="methodCategory" style="width: 100%">
              <el-option label="强化学习路径规划" value="rl" />
              <el-option label="轨迹压缩" value="compression" />
            </el-select>
          </div>

          <div class="analysis-card">
            <div class="analysis-label">预存轨迹</div>
            <el-select v-model="scenarioId" style="width: 100%" :disabled="selectedPoints.length === 2">
              <el-option v-for="scenario in presetScenarios" :key="scenario.id" :label="scenario.label" :value="scenario.id" />
            </el-select>
          </div>

          <div v-if="methodCategory === 'compression'" class="analysis-card">
            <div class="analysis-label">压缩方法</div>
            <el-select v-model="selectedCompressionMethod" style="width: 100%">
              <el-option v-for="item in compressionOptions" :key="item.value" :label="item.label" :value="item.value" />
            </el-select>
          </div>

          <div v-else class="analysis-card method-actions">
            <div class="analysis-label">强化学习方法</div>
            <el-radio-group v-model="selectedRlMethod" class="method-pill-group">
              <el-radio-button v-for="item in rlOptions" :key="item.value" :value="item.value">
                {{ item.label }}
              </el-radio-button>
            </el-radio-group>
          </div>

          <div class="analysis-card">
            <div class="analysis-label">轨迹标注</div>
            <div class="tool-action-row">
              <el-button :type="selectingPoints ? 'danger' : 'warning'" plain @click="toggleSelectingPoints">
                {{ selectingPoints ? "结束标注" : "开始标注" }}
              </el-button>
              <el-button type="primary" :disabled="!canGenerateRoute" @click="generateTrajectory">
                生成轨迹
              </el-button>
              <el-button plain @click="clearSelectedPoints">清空标注</el-button>
              <el-button v-if="canReplay" type="primary" plain @click="startPlayback()">
                重新播放 {{ selectedRlMethod.toUpperCase() }}
              </el-button>
            </div>
          </div>
        </div>

        <div class="map-panel map-panel-full">
          <div ref="mapRoot" class="map-host map-host-large"></div>
        </div>

        <div class="path-info-grid">
          <div class="analysis-card">
            <div class="analysis-label">当前视角</div>
            <div class="analysis-value compact">
              {{ methodCategory === "compression" ? selectedCompressionMethod.toUpperCase() : selectedRlMethod.toUpperCase() }}
            </div>
          </div>
          <div class="analysis-card">
            <div class="analysis-label">地图状态</div>
            <div class="analysis-value compact">{{ mapReady ? "地图已加载" : "地图加载中" }}</div>
          </div>
          <div class="analysis-card">
            <div class="analysis-label">路线状态</div>
            <div class="analysis-value compact">{{ routeStatus }}</div>
          </div>
          <div class="analysis-card">
            <div class="analysis-label">保存状态</div>
            <div class="analysis-value compact">{{ saving ? "正在写入数据库..." : saveStatus }}</div>
          </div>
          <div class="analysis-card">
            <div class="analysis-label">路线规模</div>
            <div class="analysis-value compact">
              {{ routeReady ? `${routeDistanceKm} km / ${routeDurationMin} min / ${routeGeometry.length} 个路径点` : "尚未生成" }}
            </div>
          </div>
          <div v-if="compressionStats" class="analysis-card">
            <div class="analysis-label">压缩统计</div>
            <div class="analysis-value compact">
              保留点 {{ compressionStats.kept }} / {{ compressionStats.total }}，压缩率 {{ compressionStats.ratio }}
            </div>
          </div>
        </div>

        <div class="path-bottom-grid">
          <div class="policy-note">
            <h3>当前方法的步骤解释</h3>
            <div class="route-step-list">
              <div v-for="(step, index) in currentSteps" :key="`${index}-${step.name}`" class="route-step-item">
                <div class="route-step-index">{{ index + 1 }}</div>
                <div class="route-step-copy">
                  <strong>{{ step.name }}</strong>
                  <span>{{ step.note }}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="policy-note">
            <h3>说明与图例</h3>
            <p><span class="legend-line route-base" /> Baseline：OSRM driving 推荐路线</p>
            <p><span class="legend-line dqn" /> DQN：红色离散决策链</p>
            <p><span class="legend-line ppo" /> PPO：绿色阶段策略链</p>
            <p><span class="legend-line compression-s3" /> 轨迹压缩：关键保留点及骨架连线</p>
            <p><span class="legend-dot start" /> 起点 / 终点：当前标注或预存轨迹端点</p>
          </div>
        </div>

        <div class="path-bottom-grid">
          <div class="section-block">
            <h3 class="section-title">你标注出来的点</h3>
            <div v-if="selectedPoints.length" class="selected-points-list">
              <div v-for="(point, index) in selectedPoints" :key="point.id" class="selected-point-row">
                <div class="selected-point-index">{{ index + 1 }}</div>
                <div class="selected-point-copy">
                  <strong>{{ index === 0 ? "起点" : "终点" }}</strong>
                  <span>纬度 {{ point.coords[0].toFixed(6) }}，经度 {{ point.coords[1].toFixed(6) }}</span>
                </div>
              </div>
            </div>
            <div v-else class="empty-state">{{ trajectoryHint }}</div>
          </div>

          <div class="section-block">
            <h3 class="section-title">最近保存的轨迹</h3>
            <div v-if="recentTrajectories.length" class="selected-points-list">
              <div v-for="item in recentTrajectories" :key="item.id" class="selected-point-row">
                <div class="selected-point-index">{{ item.id }}</div>
                <div class="selected-point-copy">
                  <strong>{{ item.scenarioLabel || item.trajectoryType }}</strong>
                  <span>{{ item.distanceKm }} km / {{ item.durationMin }} min / {{ item.createdAt }}</span>
                </div>
              </div>
            </div>
            <div v-else class="empty-state">数据库里还没有保存的轨迹记录。</div>
          </div>
        </div>
      </div>
    </article>
  </section>
</template>
