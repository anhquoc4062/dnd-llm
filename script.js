/* =========================================================================
   AI RPG — Character Creator
   -------------------------------------------------------------------------
   Toàn bộ dữ liệu (chủng tộc, chức nghiệp, điểm mạnh/yếu, trang bị, kỹ năng,
   vật phẩm) nằm ở file riêng: gameData.js — phải load TRƯỚC file này.

   NOTE on "Qwen": stat/equipment/skill/item randomization below is done
   locally with deterministic + weighted random pools so this page works
   standalone. If your backend already calls Qwen to generate these,
   replace the body of `randomizeLoadout()` with a fetch() to your Qwen
   endpoint (pass race, class, strengths, weaknesses, attrs) and fill
   `state.equipment / state.skills / state.items` from the response.
   ========================================================================= */

/* ---------------------------- STATE ------------------------------------ */

const state = {
  name: '',
  gender: null,
  raceId: null,
  classId: null,
  strengths: [],   // selected ids, max 2
  weaknesses: [],  // selected ids, max 2
  equipment: [],   // [{key, vi, en}]
  skills: [],      // [{key, vi, en}]
  items: [],       // [{key, vi, en}]
  campaignMode: 'ai',      // 'ai' | 'custom'
  campaignSeeds: [],       // 5 seed đầy đủ (chỉ theme được render, phần còn lại giấu khỏi UI)
  campaignSeed: null,      // seed đã chọn/khai triển — gửi kèm lúc tạo nhân vật
};

/* ---------------------------- UTIL -------------------------------------- */

function hashCode(str){
  let h = 0;
  for (let i = 0; i < str.length; i++){ h = (h << 5) - h + str.charCodeAt(i); h |= 0; }
  return Math.abs(h) || 1;
}

function seededShuffle(arr, seedStr){
  let seed = hashCode(seedStr);
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--){
    seed = (seed * 9301 + 49297) % 233280;
    const rnd = seed / 233280;
    const j = Math.floor(rnd * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function pickRandomUnique(pool, n){
  const a = [...pool];
  const out = [];
  while (a.length && out.length < n){
    const idx = Math.floor(Math.random() * a.length);
    out.push(a.splice(idx, 1)[0]);
  }
  return out;
}

function byId(id){ return document.getElementById(id); }

/* ---------------------------- RENDER: RACE / CLASS ---------------------- */

function renderCards(container, list, stateKey, onSelect){
  container.innerHTML = list.map(item => {
    const tags = ATTR_KEYS
      .filter(k => item.bonus && item.bonus[k])
      .map(k => `<span class="tag ${item.bonus[k] < 0 ? 'neg' : ''}">${k.toUpperCase()} ${item.bonus[k] > 0 ? '+' : ''}${item.bonus[k]}</span>`)
      .join('');
    return `
      <label class="opt-card" data-id="${item.id}">
        <input type="checkbox" name="${stateKey}">
        <span class="box"></span>
        <div class="opt-title">${item.icon} ${item.name}</div>
        <div class="opt-desc">${item.desc}</div>
        <div class="opt-tags">${tags}</div>
      </label>`;
  }).join('');

  container.querySelectorAll('.opt-card').forEach(card => {
    card.addEventListener('click', (e) => {
      e.preventDefault();
      const id = card.dataset.id;
      // single-select behaviour
      container.querySelectorAll('.opt-card').forEach(c => {
        c.classList.remove('selected');
        c.querySelector('input').checked = false;
      });
      card.classList.add('selected');
      card.querySelector('input').checked = true;
      onSelect(id);
    });
  });
}

/* ---------------------------- RENDER: TRAITS ----------------------------- */

function getFilteredPool(pool){
  if (!state.raceId || !state.classId) return null;
  return seededShuffle(pool, state.raceId + '::' + state.classId).slice(0, 5);
}

function renderTraits(kind){
  const isStrength = kind === 'strength';
  const container = byId(isStrength ? 'strength-container' : 'weakness-container');
  const pool = isStrength ? STRENGTH_POOL : WEAKNESS_POOL;
  const selectedArr = isStrength ? state.strengths : state.weaknesses;
  container.className = 'trait-list ' + kind;

  const options = getFilteredPool(pool);
  if (!options){
    container.innerHTML = '<div class="locked-msg">🔒 Hãy chọn chủng tộc và chức nghiệp trước.</div>';
    return;
  }

  container.innerHTML = options.map(t => {
    const checked = selectedArr.includes(t.id);
    const disabled = !checked && selectedArr.length >= 1;
    return `
      <label class="trait-card ${checked ? 'selected' : ''} ${disabled ? 'disabled' : ''}" data-id="${t.id}">
        <input type="checkbox" ${checked ? 'checked' : ''} ${disabled ? 'disabled' : ''}>
        <div class="trait-body">
          <span class="trait-name">${t.name}</span>
          <span class="trait-impact">${t.note}</span>
        </div>
      </label>`;
  }).join('');

  container.querySelectorAll('.trait-card').forEach(card => {
    card.addEventListener('click', (e) => {
      e.preventDefault();
      if (card.classList.contains('disabled')) return;
      const id = card.dataset.id;
      const idx = selectedArr.indexOf(id);
      if (idx >= 0) selectedArr.splice(idx, 1);
      else if (selectedArr.length < 1) selectedArr.push(id);
      renderTraits(kind);
      updateCounters();
      recalcAndRender();
    });
  });
}

function updateCounters(){
  byId('strength-counter').textContent = `(${state.strengths.length}/1)`;
  byId('weakness-counter').textContent = `(${state.weaknesses.length}/1)`;
}

/* ---------------------------- STAT COMPUTATION --------------------------- */

function computeStats(){
  const attrs = { str:8, dex:8, con:8, int:8, wis:8, cha:8 };
  let extraHP = 0, extraMana = 0;

  const race = RACES.find(r => r.id === state.raceId);
  const cls = CLASSES.find(c => c.id === state.classId);

  if (race) ATTR_KEYS.forEach(k => attrs[k] += race.bonus[k] || 0);
  if (cls) ATTR_KEYS.forEach(k => attrs[k] += cls.bonus[k] || 0);

  const applyEffect = (effect) => {
    ATTR_KEYS.forEach(k => { if (effect[k]) attrs[k] += effect[k]; });
    if (effect.hp) extraHP += effect.hp;
    if (effect.mana) extraMana += effect.mana;
  };
  state.strengths.forEach(id => { const t = STRENGTH_POOL.find(s => s.id === id); if (t) applyEffect(t.effect); });
  state.weaknesses.forEach(id => { const t = WEAKNESS_POOL.find(s => s.id === id); if (t) applyEffect(t.effect); });

  ATTR_KEYS.forEach(k => { attrs[k] = Math.max(1, attrs[k]); });

  const baseHP = cls ? cls.hp : 100;
  const baseMana = cls ? cls.mana : 50;
  const xpTarget = cls ? cls.xpTarget : 100;
  const manaAttr = cls ? cls.manaAttr : 'int';

  // Mốc "trung bình" của hệ thống này là 8 (điểm khởi đầu mọi stat trước khi
  // cộng race/class/trait), KHÔNG phải 10 như chuẩn D&D — dùng nhầm mốc 10 ở
  // đây từng khiến MỌI nhân vật bị trừ oan 10 HP / 6 Mana ngay từ đầu, làm các
  // class máu mỏng (vd Pháp Sư 18 HP) kết hợp chủng tộc trừ CON tụt xuống HP
  // gần 0 dù chưa vào game.
  const hp = Math.max(1, baseHP + (attrs.con - 8) * 5 + extraHP);
  const mana = Math.max(0, baseMana + (attrs[manaAttr] - 8) * 3 + extraMana);

  return { attrs, hp, mana, xpTarget };
}

/* ---------------------------- LIVE PREVIEW -------------------------------- */

function recalcAndRender(){
  const { attrs, hp, mana, xpTarget } = computeStats();

  ATTR_KEYS.forEach(k => {
    const el = byId(k);
    el.textContent = attrs[k];
    el.classList.remove('pos', 'neg', 'neutral');
    if (attrs[k] > 8) el.classList.add('pos');
    else if (attrs[k] < 8) el.classList.add('neg');
    else el.classList.add('neutral');
  });
  byId('hp').textContent = hp;
  byId('mana').textContent = mana;
  byId('xp').textContent = `0 / ${xpTarget}`;

  byId('hp-bar').style.width = '100%';
  byId('mana-bar').style.width = '100%';
  byId('xp-bar').style.width = '0%';

  const race = RACES.find(r => r.id === state.raceId);
  const cls = CLASSES.find(c => c.id === state.classId);

  byId('preview-name').textContent = state.name.trim() || 'Chưa đặt tên';
  byId('preview-class').textContent = race && cls
    ? `${race.name} · ${cls.name}${state.gender ? ' · ' + state.gender : ''}`
    : 'Chưa có chủng tộc · chức nghiệp';
  byId('avatar-race').textContent = race ? race.icon : '🧑';
  byId('avatar-class').textContent = cls ? cls.icon : '🛡️';
}

/* ---------------------------- LOADOUT RANDOMIZATION ------------------------ */

function randomizeLoadout(){
  const cls = state.classId;
  const race = state.raceId;

  const equipPool = EQUIPMENT_POOL[cls] || [];
  const equipment = pickRandomUnique(equipPool, 1);
  const traitItem = RACE_TRAIT_ITEM[race];
  if (traitItem) equipment.push(traitItem);

  const skillPool = SKILL_POOL[cls] || [];
  const skills = pickRandomUnique(skillPool, 2);

  const itemPool = [...(CLASS_ITEMS[cls] || []), ...GENERAL_ITEMS];
  const items = pickRandomUnique(itemPool, 2);

  state.equipment = equipment;
  state.skills = skills;
  state.items = items;
}

function escAttr(str){
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderLoadout(){
  // list items are {key, vi, en, desc} — hiển thị tiếng Việt chính, tiếng Anh
  // phụ bên dưới; hover hiện tooltip tuỳ chỉnh (data-tooltip, xem CSS) với mô
  // tả tiếng Việt nếu có — KHÔNG dùng title="" vì tooltip native của trình
  // duyệt có độ trễ ~1s và style OS mặc định, rất dễ bị tưởng là "không hoạt động".
  const fill = (containerId, list) => {
    const el = byId(containerId);
    el.innerHTML = list.length
      ? list.map(x => `<li${x.desc ? ` data-tooltip="${escAttr(x.desc)}"` : ''}>${x.vi}<span class="li-note">${x.en}</span></li>`).join('')
      : `<li class="empty">Chưa xác định</li>`;
  };
  fill('equipment-preview', state.equipment);
  fill('skill-preview', state.skills);
  fill('item-preview', state.items);
}

/* ---------------------------- CAMPAIGN SEED ------------------------------- */
/* Người chơi chỉ thấy "theme" (1 câu) trong dropdown/xác nhận — main_goal,
   plot, npcs, monsters, boss bị giấu, chỉ nằm trong state.campaignSeed để
   gửi thẳng lên backend lúc tạo nhân vật, không render ra DOM. */

function setCampaignStatus(msg){
  byId('campaign-status').textContent = msg || '';
}

function showChosenCampaign(theme){
  const box = byId('campaign-chosen');
  if (!theme){ box.classList.add('hidden'); return; }
  byId('campaign-chosen-theme').textContent = theme;
  box.classList.remove('hidden');
}

function setCampaignMode(mode){
  state.campaignMode = mode;
  document.querySelectorAll('#campaign-mode-group .pill').forEach(b => {
    b.classList.toggle('selected', b.dataset.mode === mode);
  });
  byId('campaign-ai-panel').classList.toggle('hidden', mode !== 'ai');
  byId('campaign-custom-panel').classList.toggle('hidden', mode !== 'custom');
  setCampaignStatus('');
}

async function generateCampaignSeeds(){
  const btn = byId('campaign-generate-btn');
  const select = byId('campaign-select');
  btn.disabled = true;
  select.disabled = true;
  state.campaignSeed = null;
  showChosenCampaign(null);
  setCampaignStatus('⏳ Đang nghĩ ra 5 kịch bản khác nhau (có thể mất một lúc)…');
  try{
    const res = await fetch('/campaign_seeds');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    state.campaignSeeds = Array.isArray(data.seeds) ? data.seeds : [];
    select.innerHTML = state.campaignSeeds.length
      ? '<option value="">— Chọn 1 kịch bản —</option>' +
        state.campaignSeeds.map((s, i) => `<option value="${i}">${s.theme}</option>`).join('')
      : '<option value="">— Không tạo được kịch bản, thử lại —</option>';
    select.disabled = false;
    setCampaignStatus(state.campaignSeeds.length ? '' : 'Không tạo được kịch bản. Hãy thử lại.');
  } catch(e){
    console.error('Không thể tạo campaign seeds:', e);
    setCampaignStatus('Lỗi kết nối server khi tạo kịch bản. Kiểm tra backend đã chạy chưa.');
  } finally {
    btn.disabled = false;
  }
}

async function expandCustomCampaign(){
  const text = byId('campaign-custom-text').value.trim();
  if (!text){ setCampaignStatus('Hãy viết ý tưởng kịch bản trước.'); return; }
  const btn = byId('campaign-expand-btn');
  btn.disabled = true;
  state.campaignSeed = null;
  showChosenCampaign(null);
  setCampaignStatus('⏳ Đang khai triển ý tưởng của bạn…');
  try{
    const res = await fetch('/campaign_seed/expand', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const seed = await res.json();
    if (seed.error) throw new Error(seed.error);
    state.campaignSeed = seed;
    showChosenCampaign(seed.theme);
    setCampaignStatus('');
  } catch(e){
    console.error('Không thể khai triển kịch bản:', e);
    setCampaignStatus('Lỗi khi khai triển kịch bản. Kiểm tra backend đã chạy chưa.');
  } finally {
    btn.disabled = false;
  }
}

function initCampaignSeed(){
  document.querySelectorAll('#campaign-mode-group .pill').forEach(btn => {
    btn.addEventListener('click', () => setCampaignMode(btn.dataset.mode));
  });
  byId('campaign-generate-btn').addEventListener('click', generateCampaignSeeds);
  byId('campaign-expand-btn').addEventListener('click', expandCustomCampaign);
  byId('campaign-select').addEventListener('change', (e) => {
    const idx = e.target.value;
    if (idx === ''){ state.campaignSeed = null; showChosenCampaign(null); return; }
    state.campaignSeed = state.campaignSeeds[Number(idx)];
    showChosenCampaign(state.campaignSeed.theme);
  });
}

/* ---------------------------- RANDOMIZE CHARACTER -------------------------- */
/* Random tên/giới tính/chủng tộc/chức nghiệp/điểm mạnh/điểm yếu. Phần chọn lựa
   (giới tính/chủng tộc/chức nghiệp/trait) tái dùng nguyên click handler đã gắn
   ở init() (cập nhật state + class 'selected' + render lại) thay vì viết lại.
   KHÔNG đụng tới kịch bản phiêu lưu (state.campaignSeed). */

const NAME_POOL = {
  'Nam': ['Kael', 'Thorin', 'Draven', 'Aldric', 'Roran', 'Garruk', 'Faelan', 'Bran', 'Cedric', 'Varek'],
  'Nữ': ['Elara', 'Isolde', 'Seraphina', 'Mira', 'Rowan', 'Thalia', 'Nyssa', 'Vesper', 'Lyra', 'Aveline'],
  'Khác': ['Ashryn', 'Quill', 'Sable', 'Nix', 'Ember', 'Vale', 'Ryn', 'Kestrel', 'Onyx', 'Sol'],
};

function randomizePlayerName(gender){
  const pool = NAME_POOL[gender] || NAME_POOL['Khác'];
  const name = pool[Math.floor(Math.random() * pool.length)];
  byId('char-name').value = name;
  state.name = name;
}

function randomizeCharacter(){
  showError('');

  const genderBtns = [...document.querySelectorAll('#gender-group .pill')];
  if (genderBtns.length) genderBtns[Math.floor(Math.random() * genderBtns.length)].click();

  randomizePlayerName(state.gender);

  const raceCards = [...document.querySelectorAll('#race-container .opt-card')];
  if (raceCards.length) raceCards[Math.floor(Math.random() * raceCards.length)].click();

  const classCards = [...document.querySelectorAll('#class-container .opt-card')];
  if (classCards.length) classCards[Math.floor(Math.random() * classCards.length)].click();

  // Trait list chỉ render sau khi CẢ race lẫn class đã chọn (xem getFilteredPool),
  // nên phải query lại strength/weakness SAU 2 click ở trên.
  const strengthCards = [...document.querySelectorAll('#strength-container .trait-card')];
  if (strengthCards.length) strengthCards[Math.floor(Math.random() * strengthCards.length)].click();

  const weaknessCards = [...document.querySelectorAll('#weakness-container .trait-card')];
  if (weaknessCards.length) weaknessCards[Math.floor(Math.random() * weaknessCards.length)].click();

  recalcAndRender();
}

/* ---------------------------- VALIDATION + SAVE --------------------------- */

function showError(msg){
  const el = byId('form-error');
  el.textContent = msg;
}

function flash(id){
  const el = byId(id);
  el.classList.remove('shake');
  void el.offsetWidth;
  el.classList.add('shake');
}

function startAdventure(){
  showError('');
  state.name = byId('char-name').value;

  if (!state.name.trim()){ showError('Hãy đặt tên cho nhân vật.'); flash('char-name'); return; }
  if (!state.gender){ showError('Hãy chọn giới tính.'); return; }
  if (!state.raceId){ showError('Hãy chọn một chủng tộc.'); return; }
  if (!state.classId){ showError('Hãy chọn một chức nghiệp.'); return; }
  if (state.strengths.length !== 1){ showError('Hãy chọn 1 điểm mạnh.'); return; }
  if (state.weaknesses.length !== 1){ showError('Hãy chọn 1 điểm yếu.'); return; }
  if (!state.campaignSeed){ showError('Hãy chọn hoặc xác nhận 1 kịch bản phiêu lưu.'); return; }

  randomizeLoadout();
  renderLoadout();
  recalcAndRender();

  const { attrs, hp, mana, xpTarget } = computeStats();
  const race = RACES.find(r => r.id === state.raceId);
  const cls = CLASSES.find(c => c.id === state.classId);

  const character = {
    name: state.name.trim(),
    gender: state.gender,
    race: race.name,
    raceEn: race.en,
    class: cls.name,
    classEn: cls.en,
    attrs,
    hp, mana,
    xp: 0,
    xpTarget,
    strengths: state.strengths.map(id => STRENGTH_POOL.find(s => s.id === id)),
    weaknesses: state.weaknesses.map(id => WEAKNESS_POOL.find(s => s.id === id)),
    equipment: state.equipment,
    skills: state.skills,
    items: state.items,
    campaignSeed: state.campaignSeed,
    createdAt: new Date().toISOString(),
  };

  // Lưu tạm vào localStorage (phòng khi không có mạng / backend chưa chạy)
  try{ localStorage.setItem('rpg_character', JSON.stringify(character)); }
  catch(e){ console.warn('Không thể lưu localStorage:', e); }

  saveToBackend(character);
}

async function saveToBackend(character){
  const badge = byId('save-badge');
  try{
    const res = await fetch('/create_character', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(character),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);

    badge.classList.remove('hidden');
    badge.innerHTML = '✅ Nhân vật đã được lưu — đang vào cuộc phiêu lưu...';
    console.log('Character saved to backend:', character);
    window.location.href = '/game';
  } catch(e){
    console.error('Không thể lưu nhân vật lên server:', e);
    showError('Đã tạo nhân vật (lưu tạm cục bộ) nhưng không kết nối được server. Kiểm tra backend đã chạy chưa.');
  }
}

/* ---------------------------- INIT ---------------------------------------- */

function init(){
  renderCards(byId('race-container'), RACES, 'race', (id) => {
    state.raceId = id;
    state.strengths = [];
    state.weaknesses = [];
    renderTraits('strength');
    renderTraits('weakness');
    updateCounters();
    recalcAndRender();
  });

  renderCards(byId('class-container'), CLASSES, 'class', (id) => {
    state.classId = id;
    state.strengths = [];
    state.weaknesses = [];
    renderTraits('strength');
    renderTraits('weakness');
    updateCounters();
    recalcAndRender();
  });

  renderTraits('strength');
  renderTraits('weakness');
  updateCounters();

  byId('char-name').addEventListener('input', (e) => {
    state.name = e.target.value;
    recalcAndRender();
  });

  document.querySelectorAll('#gender-group .pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#gender-group .pill').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      state.gender = btn.dataset.gender;
      recalcAndRender();
    });
  });

  initCampaignSeed();
  recalcAndRender();
}

document.addEventListener('DOMContentLoaded', init);