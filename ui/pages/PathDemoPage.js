const { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } = Vue;

const LONGQUANYI_QUERY = "龙泉驿区, 成都市, 四川省, 中国";
const NOMINATIM_URL = "https://nominatim.openstreetmap.org/search";

const keyPoints = [
  {
    name: "东安湖片区",
    coords: [30.6045, 104.2705],
    description: "适合展示赛事管制、临时封控和策略绕行。",
  },
  {
    name: "龙泉主城",
    coords: [30.5565, 104.2756],
    description: "适合作为中间决策节点，体现策略对主干道的偏好。",
  },
  {
    name: "洛带古镇方向",
    coords: [30.6352, 104.3495],
    description: "适合作为东侧目标点，后面你可以指定是否把它设成终点。",
  },
  {
    name: "经开区片区",
    coords: [30.5668, 104.2458],
    description: "适合作为工业区关键点，后面可接入不同奖励设计。",
  },
];

function plannerRoute(mode) {
  return mode === "PPO"
    ? [
        keyPoints[0].coords,
        [30.5902, 104.255],
        keyPoints[1].coords,
        [30.586, 104.305],
        keyPoints[2].coords,
      ]
    : [
        keyPoints[0].coords,
        [30.5902, 104.281],
        keyPoints[1].coords,
        [30.576, 104.258],
        keyPoints[3].coords,
      ];
}

export default {
  name: "PathDemoPage",
  setup() {
    const mapRoot = ref(null);
    const plannerMode = ref("DQN");
    const mapStatus = ref("正在加载龙泉驿区真实地图...");
    const mapSource = ref("");
    const mapObject = ref(null);
    const boundaryLayer = ref(null);
    const pointLayer = ref(null);
    const routeLayer = ref(null);
    const mapCenter = ref([30.56, 104.27]);
    const mapGeoJson = ref(null);
    const mapBoundingBox = ref(null);

    const modeDescription = computed(() =>
      plannerMode.value === "PPO"
        ? "PPO 视角更适合展示连续动作评估、整体路径稳定性和多步回报累计。等你给出具体起终点后，我们可以把每一步的选择和回报解释串起来。"
        : "DQN 视角更适合展示离散动作决策：每一步选哪条支路、什么时候绕行、什么时候优先走主干道。"
    );

    const ensureMap = async () => {
      await nextTick();
      if (!mapRoot.value || !window.L) return;

      if (mapObject.value) {
        mapObject.value.remove();
      }

      mapObject.value = window.L.map(mapRoot.value, {
        zoomControl: true,
        scrollWheelZoom: true,
      }).setView(mapCenter.value, 11);

      window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(mapObject.value);

      window.L.control.scale().addTo(mapObject.value);
      renderLayers();

      setTimeout(() => {
        mapObject.value?.invalidateSize(true);
      }, 120);
    };

    const renderLayers = () => {
      const map = mapObject.value;
      if (!map) return;

      if (boundaryLayer.value) {
        map.removeLayer(boundaryLayer.value);
      }
      if (pointLayer.value) {
        map.removeLayer(pointLayer.value);
      }
      if (routeLayer.value) {
        map.removeLayer(routeLayer.value);
      }

      if (mapGeoJson.value) {
        boundaryLayer.value = window.L.geoJSON(mapGeoJson.value, {
          style: {
            color: "#5c7cff",
            weight: 2,
            fillColor: "#dfe8ff",
            fillOpacity: 0.2,
          },
        }).addTo(map);
        map.fitBounds(boundaryLayer.value.getBounds(), { padding: [20, 20] });
      } else if (Array.isArray(mapBoundingBox.value) && mapBoundingBox.value.length === 4) {
        const [south, north, west, east] = mapBoundingBox.value.map(Number);
        map.fitBounds(
          [
            [south, west],
            [north, east],
          ],
          { padding: [20, 20] }
        );
      } else {
        map.setView(mapCenter.value, 11);
      }

      pointLayer.value = window.L.layerGroup();
      keyPoints.forEach((point) => {
        const marker = window.L.circleMarker(point.coords, {
          radius: 7,
          color: "#ff81ba",
          weight: 2,
          fillColor: "#ff81ba",
          fillOpacity: 0.9,
        }).bindPopup(`<strong>${point.name}</strong><br>${point.description}`);
        marker.addTo(pointLayer.value);
      });
      pointLayer.value.addTo(map);

      routeLayer.value = window.L.polyline(plannerRoute(plannerMode.value), {
        color: plannerMode.value === "PPO" ? "#45c38a" : "#5c7cff",
        weight: 5,
        opacity: 0.9,
      }).addTo(map);
    };

    const loadMap = async () => {
      mapStatus.value = "正在加载龙泉驿区真实地图...";
      mapSource.value = "";
      try {
        const url = `${NOMINATIM_URL}?format=jsonv2&polygon_geojson=1&limit=1&q=${encodeURIComponent(LONGQUANYI_QUERY)}`;
        const response = await fetch(url, { headers: { Accept: "application/json" } });
        if (!response.ok) {
          throw new Error(`地图请求失败：${response.status}`);
        }
        const payload = await response.json();
        if (!Array.isArray(payload) || !payload.length) {
          throw new Error("没有找到龙泉驿区边界");
        }
        const match = payload[0];
        mapCenter.value = [Number(match.lat), Number(match.lon)];
        mapGeoJson.value = match.geojson || null;
        mapBoundingBox.value = match.boundingbox || null;
        mapStatus.value = "龙泉驿区真实地图已加载";
        mapSource.value = "Nominatim + OpenStreetMap";
        await ensureMap();
      } catch (error) {
        mapStatus.value = `地图加载失败：${error.message}`;
      }
    };

    watch(plannerMode, () => {
      renderLayers();
      setTimeout(() => mapObject.value?.invalidateSize(true), 80);
    });

    onMounted(loadMap);
    onBeforeUnmount(() => {
      if (mapObject.value) {
        mapObject.value.remove();
        mapObject.value = null;
      }
    });

    return {
      keyPoints,
      mapRoot,
      mapSource,
      mapStatus,
      modeDescription,
      plannerMode,
    };
  },
  template: `
    <section class="page-fill page-single">
      <article class="card full-card">
        <header class="card-header">
          <div>
            <div class="eyebrow">RL Path Planning</div>
            <h2 class="card-title">路径规划演示</h2>
            <div class="card-subtitle">这里先把龙泉驿区真实地图、道路和关键点挂起来。等你给出具体路线后，我们再把 PPO / DQN 的逐步路线动画放上来。</div>
          </div>
          <div style="display:flex; gap:10px;">
            <el-button type="primary" plain @click="plannerMode = 'DQN'">切到 DQN 视角</el-button>
            <el-button type="success" plain @click="plannerMode = 'PPO'">切到 PPO 视角</el-button>
          </div>
        </header>
        <div class="card-body page-scroll">
          <div class="policy-explainer-grid">
            <div class="policy-note">
              <h3>当前这一步我们先做什么</h3>
              <p>先把龙泉驿区真实地图底图、道路和关键点立住。路线动画这一步先不乱编，等你指定起点、终点和中间约束后，我们再让 DQN / PPO 一步步走出来。</p>
            </div>
            <div class="policy-note">
              <h3>为什么这页和检索策略学习并存</h3>
              <p>路径规划页负责让人直观看懂强化学习怎样做空间决策；检索策略页负责展示强化学习怎样真实优化了当前 Agent 的知识问答与证据选择。</p>
            </div>
          </div>

          <div class="path-demo-layout">
            <div class="map-panel">
              <div ref="mapRoot" class="map-echart map-host"></div>
            </div>

            <div class="map-side">
              <div class="analysis-card">
                <div class="analysis-label">当前视角</div>
                <div class="analysis-value">{{ plannerMode }}</div>
              </div>
              <div class="analysis-card">
                <div class="analysis-label">地图状态</div>
                <div class="analysis-value compact">{{ mapStatus }}</div>
              </div>
              <div class="analysis-card">
                <div class="analysis-label">地图来源</div>
                <div class="analysis-value compact">{{ mapSource || '等待加载地图源' }}</div>
              </div>
              <div class="policy-note">
                <h3>关键点</h3>
                <p v-for="point in keyPoints" :key="point.name"><strong>{{ point.name }}</strong>：{{ point.description }}</p>
              </div>
              <div class="policy-note">
                <h3>当前说明</h3>
                <p>{{ modeDescription }}</p>
              </div>
              <div class="policy-note">
                <h3>图例</h3>
                <p><span class="legend-line dqn"></span>DQN 示例路线</p>
                <p><span class="legend-line ppo"></span>PPO 示例路线</p>
                <p><span class="legend-dot start"></span>关键点标记</p>
              </div>
            </div>
          </div>
        </div>
      </article>
    </section>
  `,
};
