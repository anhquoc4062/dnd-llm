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
  campaignHooks: [],       // 5 câu hook ngắn để CHỌN — không còn bước "xác nhận"/khai triển riêng ở đây
  campaignTheme: '',       // hook đã chọn (mode 'ai') hoặc text tự viết (mode 'custom') — gửi RAW lúc tạo
                            // nhân vật, backend tự khai triển đầy đủ Campaign Bible ở bước sau (modal tiến trình)
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
  const attrs = { str:10, dex:10, con:10, int:10, wis:10, cha:10 };
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

  // Mốc "trung bình" là 10 (chuẩn D&D 5e, modifier 0) — điểm khởi đầu mọi stat
  // trước khi cộng race/class/trait. baseHP/baseMana của mỗi class (gameData.js)
  // đã được tính cho đúng nhân vật "trung bình" này (con/mana-attr = 10).
  const hp = Math.max(1, baseHP + (attrs.con - 10) * 5 + extraHP);
  const mana = Math.max(0, baseMana + (attrs[manaAttr] - 10) * 3 + extraMana);

  return { attrs, hp, mana, xpTarget };
}

/* ---------------------------- LIVE PREVIEW -------------------------------- */

function recalcAndRender(){
  const { attrs, hp, mana, xpTarget } = computeStats();

  ATTR_KEYS.forEach(k => {
    const el = byId(k);
    el.textContent = attrs[k];
    el.classList.remove('pos', 'neg', 'neutral');
    if (attrs[k] > 10) el.classList.add('pos');
    else if (attrs[k] < 10) el.classList.add('neg');
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
  renderRaceAvatar(byId('avatar-race'), race, state.gender);
  renderClassAvatar(byId('avatar-class'), cls);
}

/* Ảnh chân dung theo tộc + giới tính (nếu có) — fallback về emoji icon khi
   chưa chọn tộc/giới tính hoặc tộc đó chưa có ảnh (vd tộc tuỳ biến). */
function renderRaceAvatar(el, race, gender){
  const src = race && (gender === 'Nữ' ? race.avatarFemale : race.avatarMale);
  if (src){
    el.innerHTML = `<img src="${src}" alt="${race.name}">`;
  } else {
    el.textContent = race ? race.icon : '🧑';
  }
}

/* Huy hiệu chức nghiệp (nếu có ảnh) — fallback về emoji icon. */
function renderClassAvatar(el, cls){
  if (cls && cls.avatar){
    el.innerHTML = `<img src="${cls.avatar}" alt="${cls.name}">`;
  } else {
    el.textContent = cls ? cls.icon : '🛡️';
  }
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
/* Người chơi chỉ CHỌN 1 hook (hoặc viết ý tưởng riêng) — không còn bước
   "xác nhận kịch bản"/khai triển riêng ở màn này. state.campaignTheme giữ
   nguyên văn (raw) hook đã chọn/text đã gõ, gửi thẳng lên backend lúc bấm
   "Khởi tạo nhân vật" — backend tự khai triển đầy đủ Campaign Bible trong
   lúc hiện modal tiến trình (xem showSetupModal()). */

function setCampaignStatus(msg){
  byId('campaign-status').textContent = msg || '';
}

function setCampaignMode(mode){
  state.campaignMode = mode;
  document.querySelectorAll('#campaign-mode-group .pill').forEach(b => {
    b.classList.toggle('selected', b.dataset.mode === mode);
  });
  byId('campaign-ai-panel').classList.toggle('hidden', mode !== 'ai');
  byId('campaign-custom-panel').classList.toggle('hidden', mode !== 'custom');
}

async function generateCampaignSeeds(){
  const btn = byId('campaign-generate-btn');
  const select = byId('campaign-select');
  btn.disabled = true;
  select.disabled = true;
  state.campaignTheme = '';
  setCampaignStatus('⏳ Đang nghĩ ra 5 câu mở đầu…');
  try{
    const res = await fetch('/campaign_hooks');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    state.campaignHooks = Array.isArray(data.hooks) ? data.hooks : [];
    select.innerHTML = state.campaignHooks.length
      ? '<option value="">— Chọn 1 kịch bản —</option>' +
        state.campaignHooks.map((h, i) => `<option value="${i}">${h}</option>`).join('')
      : '<option value="">— Không tạo được kịch bản, thử lại —</option>';
    select.disabled = false;
    setCampaignStatus(state.campaignHooks.length ? 'Chọn 1 kịch bản.' : 'Không tạo được kịch bản. Hãy thử lại.');
  } catch(e){
    console.error('Không thể tạo campaign hooks:', e);
    setCampaignStatus('Lỗi kết nối server khi tạo kịch bản. Kiểm tra backend đã chạy chưa.');
  } finally {
    btn.disabled = false;
  }
}

function initCampaignSeed(){
  document.querySelectorAll('#campaign-mode-group .pill').forEach(btn => {
    btn.addEventListener('click', () => setCampaignMode(btn.dataset.mode));
  });
  byId('campaign-generate-btn').addEventListener('click', generateCampaignSeeds);
  byId('campaign-select').addEventListener('change', (e) => {
    const idx = e.target.value;
    state.campaignTheme = idx === '' ? '' : (state.campaignHooks[Number(idx)] || '');
  });
  byId('campaign-custom-text').addEventListener('input', (e) => {
    state.campaignTheme = e.target.value.trim();
  });
}

/* ---------------------------- RANDOMIZE CHARACTER -------------------------- */
/* Random tên/giới tính/chủng tộc/chức nghiệp/điểm mạnh/điểm yếu. Phần chọn lựa
   (giới tính/chủng tộc/chức nghiệp/trait) tái dùng nguyên click handler đã gắn
   ở init() (cập nhật state + class 'selected' + render lại) thay vì viết lại.
   KHÔNG đụng tới kịch bản phiêu lưu (state.campaignTheme). */

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
  if (!state.campaignTheme.trim()){ showError('Hãy chọn hoặc viết 1 kịch bản phiêu lưu.'); return; }

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
    campaignTheme: state.campaignTheme.trim(),
    campaignMode: state.campaignMode,
    createdAt: new Date().toISOString(),
  };

  // Lưu tạm vào localStorage (phòng khi không có mạng / backend chưa chạy)
  try{ localStorage.setItem('rpg_character', JSON.stringify(character)); }
  catch(e){ console.warn('Không thể lưu localStorage:', e); }

  saveToBackend(character);
}

async function saveToBackend(character){
  try{
    const res = await fetch('/create_character', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(character),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // Nhân vật đã tạo xong — backend giờ chạy NỀN 3 bước (sinh Bible -> sinh
    // milestone 1 -> DM mở màn). Mở modal tiến trình, poll /setup_status tới
    // khi "ready" mới điều hướng sang /game (không redirect ngay như trước).
    console.log('Character created, campaign setup started:', character);
    openSetupModal();
    pollSetupStatus();
  } catch(e){
    console.error('Không thể lưu nhân vật lên server:', e);
    showError('Không thể tạo nhân vật — không kết nối được server. Kiểm tra backend đã chạy chưa.');
  }
}

/* ---------------------------- MODAL TIẾN TRÌNH TẠO CAMPAIGN --------------- */
/* Bấm "Khởi tạo nhân vật" xong là hiện modal này ngay — sinh Bible/milestone
   1/DM mở màn đều chạy NỀN ở backend (asyncio.create_task, xem main.py:
   run_campaign_setup), frontend chỉ poll /setup_status mỗi ~1.5s để biết
   đang ở bước nào, tạo cảm giác "đang dựng thế giới" thay vì màn hình trắng
   chờ đợi không rõ lý do. */

const SETUP_STAGE_ORDER = ['bible', 'milestone', 'opening', 'ready'];
let setupPollTimer = null;

function openSetupModal(){
  byId('setup-modal-overlay').classList.remove('hidden');
  byId('setup-error-text').classList.add('hidden');
  byId('setup-retry-btn').classList.add('hidden');
  updateSetupModal('bible');
}

function updateSetupModal(stage){
  const currentIdx = SETUP_STAGE_ORDER.indexOf(stage);
  ['bible', 'milestone', 'opening'].forEach((s) => {
    const el = byId('setup-step-' + s);
    const idx = SETUP_STAGE_ORDER.indexOf(s);
    el.classList.toggle('done', currentIdx > idx);
    el.classList.toggle('active', currentIdx === idx);
  });
}

async function pollSetupStatus(){
  if (setupPollTimer){ clearTimeout(setupPollTimer); setupPollTimer = null; }
  let data;
  try{
    const res = await fetch('/setup_status');
    data = await res.json();
  } catch(e){
    console.error('Lỗi khi kiểm tra tiến trình tạo campaign:', e);
    setupPollTimer = setTimeout(pollSetupStatus, 1500);
    return;
  }

  if (data.stage === 'error'){
    byId('setup-error-text').textContent = '⚠️ ' + (data.error || 'Đã có lỗi khi dựng thế giới.');
    byId('setup-error-text').classList.remove('hidden');
    byId('setup-retry-btn').classList.remove('hidden');
    return;
  }

  if (data.stage === 'ready'){
    updateSetupModal('ready');
    window.location.href = '/game';
    return;
  }

  updateSetupModal(data.stage || 'bible');
  setupPollTimer = setTimeout(pollSetupStatus, 1500);
}

function initSetupModal(){
  byId('setup-retry-btn').addEventListener('click', () => {
    byId('setup-modal-overlay').classList.add('hidden');
    startAdventure();
  });
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
  initSetupModal();
  recalcAndRender();
}

document.addEventListener('DOMContentLoaded', init);