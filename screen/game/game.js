/* ============================================================
   AI RPG — Game Screen
   Loads the character sheet from /character_info and drives the
   Dungeon Master conversation through /start_game and /chat.
   ============================================================ */

function byId(id){ return document.getElementById(id); }

/* RACES/CLASSES đến từ game-data.js (nguồn dữ liệu chung với màn tạo nhân
   vật) — tra icon theo tên tiếng Việt lưu trong DB, thay vì map riêng dễ lệch
   khi thêm chủng tộc/chức nghiệp mới. */
function findIcon(list, name, fallback){
  const item = (list || []).find(x => x.name === name);
  return item ? item.icon : fallback;
}

function findRace(name){
  return (typeof RACES !== 'undefined' ? RACES : []).find(x => x.name === name);
}

function findClass(name){
  return (typeof CLASSES !== 'undefined' ? CLASSES : []).find(x => x.name === name);
}

/* Ảnh chân dung theo tộc + giới tính (nếu có) — fallback về emoji icon khi
   tộc đó chưa có ảnh (vd tộc tuỳ biến). */
function renderRaceAvatar(el, race, gender, fallbackIcon){
  const src = race && (gender === 'Nữ' ? race.avatarFemale : race.avatarMale);
  if (src){
    el.innerHTML = `<img src="${src}" alt="${race.name}">`;
  } else {
    el.textContent = race ? race.icon : fallbackIcon;
  }
}

/* Huy hiệu chức nghiệp (nếu có ảnh) — fallback về emoji icon. */
function renderClassAvatar(el, cls, fallbackIcon){
  if (cls && cls.avatar){
    el.innerHTML = `<img src="${cls.avatar}" alt="${cls.name}">`;
  } else {
    el.textContent = cls ? cls.icon : fallbackIcon;
  }
}

let isGameOver = false;

/* Bubble user/dm của lượt GẦN NHẤT — backend chỉ giữ 1 snapshot (undo 1 cấp),
   nên nút "Thử lại" chỉ có ý nghĩa trên lượt mới nhất. */
let lastTurnUserBubble = null;
let lastTurnDmBubble = null;

/* ---------------------------- Character sheet ---------------------------- */

async function loadCharacter(){
  try{
    const res = await fetch('/character_info');
    const char = await res.json();
    if (!char || !char.name){
      byId('char-name').textContent = 'Chưa có nhân vật';
      byId('char-subtitle').textContent = 'Hãy tạo nhân vật trước';
      return null;
    }
    renderCharacter(char);
    return char;
  } catch(e){
    console.error('Không thể tải thông tin nhân vật:', e);
    byId('char-subtitle').textContent = 'Lỗi kết nối server';
    return null;
  }
}

function fillList(containerId, items, withNote){
  const el = byId(containerId);
  if (!items || items.length === 0){
    el.innerHTML = '<li class="empty">Không có</li>';
    return;
  }
  el.innerHTML = items.map(item => {
    if (withNote && typeof item === 'object'){
      return `<li>${item.name}<span class="li-note">${item.note || ''}</span></li>`;
    }

    // equipment / skills / items: {key, vi, en, desc}
    if (typeof item === 'object'){
      const viName = item.vi || item.name || '';
      const enName = item.en || '';
      const showEn = enName && enName.toLowerCase() !== viName.toLowerCase();
      const tooltipAttr = item.desc ? ` data-tooltip="${escAttr(item.desc)}"` : '';
      return `<li class="${item.cooldown_current ? 'cooldown' : ''}"${tooltipAttr}>${escapeHtml(viName)} ${item.cooldown_current ? `(Còn ${item.cooldown_current} lượt)` : ''} ${showEn ? `<span class="li-note-en">${escapeHtml(enName)}</span>` : ''}</li>`;
    }
    return `<li>${escapeHtml(item)}</li>`;
  }).join('');
}

function renderCharacter(char){
  renderRaceAvatar(byId('avatar-race'), findRace(char.race), char.gender, '🧑');
  renderClassAvatar(byId('avatar-class'), findClass(char.character_class), '🛡️');
  byId('char-name').textContent = char.name || '-';
  byId('char-subtitle').textContent = `${char.race || '-'} · ${char.character_class || '-'} · ${char.gender || '-'}`;

  byId('char-level').textContent = char.level ?? 1;
  byId('char-gold').textContent = char.gold ?? 0;
  byId('char-exp').textContent = `${char.xp ?? 0} / ${char.xp_target ?? 100}`;
  const xpPct = char.xp_target ? Math.min(100, (char.xp / char.xp_target) * 100) : 0;
  byId('xp-bar').style.width = xpPct + '%';

  const hp = char.hp ?? 100, maxHp = char.max_hp ?? 100;
  byId('hp-text').textContent = `${hp} / ${maxHp}`;
  byId('hp-bar').style.width = Math.max(0, Math.min(100, (hp / maxHp) * 100)) + '%';

  const mana = char.mana ?? 50, maxMana = char.max_mana ?? 50;
  byId('mana-text').textContent = `${mana} / ${maxMana}`;
  byId('mana-bar').style.width = Math.max(0, Math.min(100, (mana / maxMana) * 100)) + '%';

  const attrs = char.attrs || {};
  ['str','dex','con','int','wis','cha'].forEach(k => {
    const el = byId('stat-' + k);
    if (!el) return;
    const val = attrs[k] ?? 10;
    el.textContent = val;
    el.classList.remove('pos', 'neg', 'neutral');
    if (val > 10) el.classList.add('pos');
    else if (val < 10) el.classList.add('neg');
    else el.classList.add('neutral');
  });

  fillList('strengths-list', char.strengths, true);
  fillList('weaknesses-list', char.weaknesses, true);
  fillList('equipment-list', char.equipment, false);
  fillList('skills-list', char.skills, false);
  fillList('items-list', char.items, false);
}

/* ---------------------------- Chat ---------------------------- */

function addMessage(role, text){
  const container = byId('chat-container');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  container.appendChild(div);
  setTimeout(() => {
    container.scrollTop = container.scrollHeight;
  }, 100);
  return div;
}

function addLoading(){
  const container = byId('chat-container');
  const div = document.createElement('div');
  div.className = 'msg loading';
  div.innerHTML = 'Người kể chuyện đang suy nghĩ<span class="dot-loading"><span></span><span></span><span></span></span>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

/**
 * Render 4 lựa chọn kèm badge ADV/DIS, DC (nếu cần roll) và lý do (điểm mạnh/điểm yếu liên quan).
 * choices: [{ text, needs_roll: bool, roll: 'normal'|'advantage'|'disadvantage', dc: number|null, reason: {type, name} | null }]
 */
function renderChoices(choices) {
    const container = byId('choices-container');

    if (!choices || choices.length === 0) {
        container.innerHTML = '';
        return;
    }

    const rollLabel = {
        advantage: 'ADV',
        disadvantage: 'DIS',
        normal: ''
    };

    const rollClass = {
        advantage: 'choice-adv',
        disadvantage: 'choice-dis',
        normal: 'choice-normal'
    };

    container.innerHTML = choices.map(choice => {

        const badge =
            choice.roll === 'normal'
                ? ''
                : `<span class="choice-roll ${rollClass[choice.roll]}">${rollLabel[choice.roll]}</span>`;

        const rollInfo = choice.needs_roll
            ? `<span class="choice-dc">🎲 Cần roll${choice.dc != null ? ` · DC ${choice.dc}` : ''}</span>`
            : `<span class="choice-dc choice-dc-auto">✔️ Không cần roll</span>`;

        const reason =
            choice.reason
                ? `<div class="choice-reason reason-${choice.reason.type}">
                        ${escapeHtml(choice.reason.type.toUpperCase())}: ${escapeHtml(choice.reason.name)}
                   </div>`
                : '';

        const header = `
            <div class="choice-header">
                ${badge}
                ${rollInfo}
            </div>
        `;

        return `
            <button class="choice-btn ${rollClass[choice.roll]}"
                    data-text="${escapeHtml(choice.text)}">

                ${header}

                <div class="choice-text">
                    ${escapeHtml(choice.text)}
                </div>

                ${reason}

            </button>
        `;
    }).join('');

    container.querySelectorAll('.choice-btn').forEach(btn => {
        btn.onclick = () => handlePlayerAction(btn.dataset.text);
    });
}

function escapeHtml(str){
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/* Tên item/skill "thật" — loại bỏ rỗng, null, và chuỗi rác "null"/"none" mà
   backend đôi khi lỡ đưa vào (tránh render "Sử dụng: null"). */
function isRealName(name){
  const s = (name == null ? '' : String(name)).trim().toLowerCase();
  return s !== '' && s !== 'null' && s !== 'none' && s !== 'undefined';
}

/* Riêng cho attribute HTML (vd title="...") — escapeHtml() ở trên KHÔNG escape
   dấu " vì nó chỉ escape đúng cho text node, không an toàn khi nhét vào giữa
   một attribute value có dấu ngoặc kép. */
function escAttr(str){
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function setInputsEnabled(enabled){
  byId('custom-action').disabled = !enabled;
  byId('send-btn').disabled = !enabled;
  byId('choices-container').querySelectorAll('.choice-btn').forEach(b => b.disabled = !enabled);
}

/**
 * 1 lượt hành động giờ tách 3 bước:
 * 1. /chat/classify — xác định có cần roll hay không (không tung xúc xắc).
 * 2. /chat/roll — tung xúc xắc thật, tiêu hao tài nguyên (NHANH, không gọi
 *    LLM) -> kết quả thành/bại hiện NGAY, không phải chờ AI kể chuyện.
 * 3. /chat/narrate — gọi LLM kể chuyện dựa trên kết quả đã roll, chỉ chạy
 *    SAU KHI người chơi đã thấy xong kết quả roll (bấm "Tiếp tục").
 * needs_roll=false -> không có popup, vẫn roll (nhanh, không hiện gì) rồi
 * narrate ngay. needs_roll=true -> mở popup, bấm "Tung xúc xắc" mới roll.
 */
async function handlePlayerAction(text) {
    if (!text || !text.trim()) return;
    setInputsEnabled(false);
    byId('choices-container').innerHTML = '';
    // Bắt đầu lượt mới -> lượt trước không còn là "gần nhất" nữa, gỡ nút thử
    // lại cũ đi (backend chỉ giữ đúng 1 snapshot, retry nút cũ giờ sẽ retry
    // NHẦM sang lượt vừa gửi này chứ không phải lượt nó đang đứng trên).
    const oldRetryBtn = byId('chat-container').querySelector('.retry-btn');
    if (oldRetryBtn) oldRetryBtn.remove();

    const userBubble = addMessage('user', text);
    const loadingEl = addLoading();

    try {
        const res = await fetch('/chat/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });
        const classifyData = await res.json();

        if (classifyData.error) {
            loadingEl.remove();
            addMessage('dm', `⚠️ ${classifyData.error}`);
            setInputsEnabled(true);
            return;
        }

        if (!classifyData.needs_roll) {
            const rollData = await fetchRoll();
            if (rollData.error) {
                loadingEl.remove();
                addMessage('dm', `⚠️ ${rollData.error}`);
                setInputsEnabled(true);
                return;
            }
            const data = await fetchNarrate();
            loadingEl.remove();
            finishTurn(data, userBubble);
        } else {
            loadingEl.remove();
            openDiceModal(classifyData, userBubble);
        }
    } catch (e) {
        console.error('Lỗi:', e);
        loadingEl.remove();
        addMessage('dm', '⚠️ Hệ thống gặp lỗi xử lý.');
        setInputsEnabled(true);
    }
}

/** /chat/roll — tung xúc xắc thật, trả kết quả NGAY (không có story, không
 * gọi LLM): {success, roll_type, dice, dc, target_ac, attribute}. */
async function fetchRoll(){
    const res = await fetch('/chat/roll', { method: 'POST' });
    return await res.json();
}

/** /chat/narrate — gọi LLM kể chuyện dựa trên kết quả /chat/roll đã lưu,
 * trả kết quả đầy đủ {story, mechanics, choices}, y hệt format /chat cũ. */
async function fetchNarrate(){
    const res = await fetch('/chat/narrate', { method: 'POST' });
    return await res.json();
}

/** Hoàn tất 1 lượt: render story/choices vào khung chat, cập nhật chỉ số,
 * mở lại input. Dùng chung cho nhánh có popup lẫn không popup. */
function finishTurn(data, userBubble){
    if (data.error) {
        addMessage('dm', `⚠️ ${data.error}`);
        setInputsEnabled(true);
        return;
    }

    const dmBubble = renderDmTurn(data);
    lastTurnUserBubble = userBubble;
    lastTurnDmBubble = dmBubble;

    renderChoices(data.choices);

    if (data.mechanics && (data.mechanics.is_dead || data.mechanics.character_died)) {
        showGameOver();
    }

    renderEntities(data.active_entities);
    if (data.act_index !== undefined) updateActLabel(data.act_index);

    loadCharacter();
    fetchAndRenderContext();
    setInputsEnabled(true);
}

/* ---------------------------- Context panel (ảnh scene) ---------------------------- */

let contextPollTimer = null;

/** Cập nhật panel context (ảnh + tên + mô tả location/quái/NPC hiện tại).
 * ctx: {kind, name, description, image_path} — mọi field có thể null nếu
 * chưa có gì được ghi nhận. */
function renderContext(ctx){
    const img = byId('context-image');
    const placeholder = byId('context-placeholder');
    const empty = byId('context-empty');
    const nameEl = byId('context-name');
    const descEl = byId('context-desc');

    renderMilestone(ctx);
    if (ctx) updateActLabel(ctx.act_index, ctx.act_title);

    if (!ctx || !ctx.name) {
        img.classList.add('hidden');
        placeholder.classList.add('hidden');
        nameEl.classList.add('hidden');
        descEl.classList.add('hidden');
        empty.classList.remove('hidden');
        return;
    }

    empty.classList.add('hidden');
    nameEl.textContent = ctx.name;
    nameEl.classList.remove('hidden');
    descEl.textContent = ctx.description || '';
    descEl.classList.toggle('hidden', !ctx.description);

    if (ctx.image_path) {
        img.src = ctx.image_path;
        img.classList.remove('hidden');
        placeholder.classList.add('hidden');
    } else {
        // Ảnh đang generate bất đồng bộ ở backend — hiện placeholder, việc
        // poll tiếp có/không do fetchAndRenderContext() quyết định.
        img.classList.add('hidden');
        placeholder.classList.remove('hidden');
    }
}

/** Milestone hiện tại — hiện giữa ảnh cảnh và danh sách NPC/quái, tiếng Anh
 * (lấy thẳng từ Bible/milestone, chỉ để người chơi dễ TRACKING đang làm gì,
 * không phải lời thoại trong truyện). */
function renderMilestone(ctx){
    const panel = byId('milestone-panel');
    if (!panel) return;
    if (!ctx || !ctx.milestone_title){
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');
    byId('milestone-title').textContent = ctx.milestone_title;
    byId('milestone-objective').textContent = ctx.milestone_objective || '';
}

/** Vẽ danh sách NPC/quái đang có trong cảnh kèm thanh máu — trước đây backend
 * track HP quái đúng nhưng UI không hiện, người chơi tưởng đánh "vô dụng". */
function renderEntities(list){
    const panel = byId('entities-panel');
    const ul = byId('entities-list');
    if (!panel || !ul) return;
    const items = (list || []).filter(e => e && e.name);
    if (items.length === 0){
        panel.classList.add('hidden');
        ul.innerHTML = '';
        return;
    }
    panel.classList.remove('hidden');
    ul.innerHTML = items.map(e => {
        const max = Math.max(1, e.max_hp || 1);
        const hp = Math.max(0, Math.min(max, e.hp == null ? max : e.hp));
        const pct = (hp / max) * 100;
        const icon = e.hostile ? '👹' : (e.type === 'monster' ? '🐾' : '🧑');
        const cls = e.hostile ? 'ent-hostile' : 'ent-friendly';
        const badge = e.image_path
            ? `<img src="${e.image_path}" alt="${escapeHtml(e.name)}">`
            : icon;
        return `<li class="ent-row ${cls}">`
            + `<div class="ent-badge">${badge}</div>`
            + `<div class="ent-body">`
            + `<div class="ent-head"><span class="ent-name">${escapeHtml(e.name)}</span>`
            + `<span class="ent-hp">${hp}/${max}</span></div>`
            + `<div class="ent-bar"><div class="ent-bar-fill" style="width:${pct}%"></div></div>`
            + `</div></li>`;
    }).join('');
}

/** Nhãn "Chương" trên header, suy từ act_index (0/1/2 -> I/II/III), kèm tên
 * act (purpose lấy từ Campaign Bible, tiếng Anh) nếu có. actTitle chỉ đến từ
 * /scene_context (renderContext) — các nơi khác chỉ có act_index nên gọi
 * updateActLabel(actIndex) trơn, hàm tự nhớ lại title cũ (_lastActTitle) thay
 * vì xoá mất chữ đã hiện trước đó. Trước đây bị hardcode "Chương II" nên sai
 * ngay từ đầu game. */
let _lastActTitle = null;
function updateActLabel(actIndex, actTitle){
    const el = byId('act-label');
    if (!el) return;
    if (actTitle !== undefined) _lastActTitle = actTitle;
    const roman = ['I', 'II', 'III'][Math.max(0, Math.min(2, actIndex || 0))] || 'I';
    el.textContent = _lastActTitle ? `Chương ${roman} · ${_lastActTitle}` : `Chương ${roman} · Cuộc Phiêu Lưu`;
}

/** Lấy context hiện tại từ backend, render ngay (tên/mô tả luôn có sẵn đồng
 * bộ), rồi nếu ảnh chưa xong thì tự poll thêm vài lần tới khi có ảnh hoặc
 * context đã bị lượt mới ghi đè (name đổi) thì dừng — ảnh generate bất đồng
 * bộ ở backend, không chặn lượt chơi nào cả. */
async function fetchAndRenderContext(){
    if (contextPollTimer) {
        clearTimeout(contextPollTimer);
        contextPollTimer = null;
    }
    let ctx;
    try {
        const res = await fetch('/scene_context');
        ctx = await res.json();
    } catch (e) {
        console.error('Lỗi khi tải context:', e);
        return;
    }
    renderContext(ctx);

    if (ctx && ctx.name && !ctx.image_path) {
        pollContextImage(ctx.name, 0);
    }
}

function pollContextImage(expectedName, attempt){
    // ~90s tối đa — ảnh SD thường xong trong 1-2s (model đã load sẵn trong bộ
    // nhớ), nhưng LẦN GENERATE ĐẦU TIÊN sau khi backend khởi động phải lazy-
    // load model trước (torch/diffusers import + tải checkpoint vào VRAM),
    // có thể mất 15-40s — cần đủ thời gian chờ cho riêng lần đầu đó.
    if (attempt >= 45) return;
    contextPollTimer = setTimeout(async () => {
        let ctx;
        try {
            const res = await fetch('/scene_context');
            ctx = await res.json();
        } catch (e) {
            return;
        }
        if (!ctx || ctx.name !== expectedName) return; // context đã đổi, dừng poll cũ
        renderContext(ctx);
        if (!ctx.image_path) pollContextImage(expectedName, attempt + 1);
    }, 2000);
}

/* ---------------------------- Popup tung xúc xắc ---------------------------- */

const ROLL_TYPE_LABEL = { advantage: 'LỢI THẾ', disadvantage: 'BẤT LỢI', normal: 'THƯỜNG' };
const ROLL_TYPE_CLASS = { advantage: 'choice-adv', disadvantage: 'choice-dis', normal: '' };

function openDiceModal(classifyData, userBubble){
    const rollType = classifyData.roll_type || 'normal';
    const rollClass = ROLL_TYPE_CLASS[rollType] || '';

    const badge = byId('dice-roll-badge');
    badge.className = 'choice-roll' + (rollClass ? ' ' + rollClass : '');
    badge.textContent = ROLL_TYPE_LABEL[rollType] || 'THƯỜNG';

    const dcLabel = byId('dice-dc-label');
    if (classifyData.contest_type === 'attack') {
        dcLabel.textContent = '🎯 Cần đánh trúng mục tiêu';
    } else if (classifyData.dc != null) {
        dcLabel.textContent = `DC ${classifyData.dc}`;
    } else {
        dcLabel.textContent = '';
    }

    const die1 = byId('die-1');
    const die2 = byId('die-2');
    die1.className = 'die';
    die1.textContent = '?';
    die2.className = rollType === 'normal' ? 'die hidden' : 'die';
    die2.textContent = '?';

    byId('dice-result').classList.add('hidden');
    const rollBtn = byId('dice-roll-btn');
    rollBtn.classList.remove('hidden');
    rollBtn.disabled = false;
    rollBtn.textContent = '🎲 Tung xúc xắc';
    byId('dice-continue-btn').classList.add('hidden');

    byId('dice-modal-overlay').classList.remove('hidden');

    rollBtn.onclick = () => performRoll(userBubble);
}

/** Bấm "Tung xúc xắc" -> chỉ gọi /chat/roll (nhanh, không có LLM) -> hiện
 * thành/bại NGAY trong popup. LLM kể chuyện (/chat/narrate) chỉ chạy SAU khi
 * người chơi đã xem xong kết quả và bấm "Tiếp tục". */
async function performRoll(userBubble){
    const rollBtn = byId('dice-roll-btn');
    const die1 = byId('die-1');
    const die2 = byId('die-2');

    rollBtn.disabled = true;
    rollBtn.textContent = 'Đang tung…';
    die1.classList.add('spinning');
    if (!die2.classList.contains('hidden')) die2.classList.add('spinning');

    // Chờ tối thiểu 700ms trước khi hiện kết quả, dù backend trả về nhanh hơn
    // (nếu không, animation xoay sẽ bị "chớp" mất, mất cảm giác đang tung).
    const minSpin = new Promise(resolve => setTimeout(resolve, 700));
    let rollData;
    try {
        [rollData] = await Promise.all([fetchRoll(), minSpin]);
    } catch (e) {
        console.error('Lỗi khi tung xúc xắc:', e);
        closeDiceModal();
        addMessage('dm', '⚠️ Hệ thống gặp lỗi xử lý.');
        setInputsEnabled(true);
        return;
    }

    die1.classList.remove('spinning');
    die2.classList.remove('spinning');

    if (rollData.error) {
        closeDiceModal();
        addMessage('dm', `⚠️ ${rollData.error}`);
        setInputsEnabled(true);
        return;
    }

    revealDiceResult(rollData);
    byId('dice-continue-btn').onclick = () => continueAfterRoll(userBubble);
}

/** Người chơi đã xem xong kết quả roll, bấm "Tiếp tục" -> đóng popup, gọi
 * /chat/narrate (LLM kể chuyện) rồi render lượt DM như cũ. */
async function continueAfterRoll(userBubble){
    closeDiceModal();
    const loadingEl = addLoading();
    try {
        const data = await fetchNarrate();
        loadingEl.remove();
        finishTurn(data, userBubble);
    } catch (e) {
        console.error('Lỗi khi kể chuyện:', e);
        loadingEl.remove();
        addMessage('dm', '⚠️ Hệ thống gặp lỗi xử lý.');
        setInputsEnabled(true);
    }
}

function revealDiceResult(mechanics){
    const dice = mechanics.dice;
    const die1 = byId('die-1');
    const die2 = byId('die-2');
    const detail = byId('dice-result-detail');

    if (dice) {
        const rolls = dice.rolls || [];
        die1.classList.remove('hidden');
        die1.textContent = rolls[0] ?? '?';

        if (rolls.length > 1) {
            const firstTaken = rolls[0] === dice.taken;
            die2.classList.remove('hidden');
            die2.textContent = rolls[1];
            die1.classList.toggle('taken', firstTaken);
            die1.classList.toggle('discarded', !firstTaken);
            die2.classList.toggle('taken', !firstTaken);
            die2.classList.toggle('discarded', firstTaken);
        } else {
            die2.classList.add('hidden');
        }

        const target = mechanics.target_ac != null
            ? `AC ${mechanics.target_ac}`
            : (mechanics.dc != null ? `DC ${mechanics.dc}` : '');
        detail.textContent = `Xúc xắc: ${dice.taken} + ${dice.modifier} = ${dice.total}` + (target ? ` (Cần: ${target})` : '');
    } else {
        // forced_fail (hết item/cooldown/mana...) -> không có xúc xắc thật để hiện.
        die1.classList.add('hidden');
        die2.classList.add('hidden');
        detail.textContent = '';
    }

    const banner = byId('dice-result-banner');
    banner.textContent = mechanics.success ? '✅ Thành công' : '❌ Thất bại';
    banner.className = 'dice-result-banner ' + (mechanics.success ? 'success' : 'fail');

    byId('dice-result').classList.remove('hidden');
    byId('dice-roll-btn').classList.add('hidden');
    byId('dice-continue-btn').classList.remove('hidden');
}

function closeDiceModal(){
    byId('dice-modal-overlay').classList.add('hidden');
}

/**
 * Thử lại lượt /chat gần nhất — khôi phục state (HP/mana/gold/xp/entities/
 * loot/history) về NGAY TRƯỚC lượt đó rồi gọi lại backend với CÙNG hành động,
 * thay bubble user+dm cũ bằng kết quả mới. Chỉ áp dụng cho lượt mới nhất
 * (backend chỉ giữ 1 snapshot).
 */
async function retryLastTurn(){
    if (!lastTurnDmBubble) return;
    const wasGameOver = isGameOver;
    setInputsEnabled(false);
    byId('choices-container').innerHTML = '';

    const retryBtn = lastTurnDmBubble.querySelector('.retry-btn');
    if (retryBtn){ retryBtn.disabled = true; retryBtn.textContent = '⏳'; }

    try {
        const res = await fetch('/chat/retry', { method: 'POST' });
        const data = await res.json();

        if (data.error) {
            addMessage('dm', `⚠️ ${data.error}`);
            return;
        }

        if (lastTurnUserBubble) lastTurnUserBubble.remove();
        if (lastTurnDmBubble) lastTurnDmBubble.remove();

        // Lượt cũ từng khiến chết -> gỡ banner, mở lại input trước khi render lượt mới
        if (wasGameOver) {
            const banner = byId('chat-container').querySelector('.death-banner');
            if (banner) banner.remove();
            isGameOver = false;
            byId('custom-action').placeholder = 'Nhập hành động của riêng bạn…';
        }

        const newUserBubble = addMessage('user', data.replayed_input || '');
        const newDmBubble = renderDmTurn(data);
        lastTurnUserBubble = newUserBubble;
        lastTurnDmBubble = newDmBubble;

        renderChoices(data.choices);

        if (data.mechanics && (data.mechanics.is_dead || data.mechanics.character_died)) {
            showGameOver();
        }

        renderEntities(data.active_entities);
        if (data.act_index !== undefined) updateActLabel(data.act_index);

        loadCharacter();
        fetchAndRenderContext();
    } catch (e) {
        console.error('Lỗi khi thử lại:', e);
        addMessage('dm', '⚠️ Không thể thử lại lượt này.');
    } finally {
        setInputsEnabled(true);
    }
}

function sendCustomAction(){
  const input = byId('custom-action');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  handlePlayerAction(text);
}

/* ---------------------------- Init ---------------------------- */

async function startStory() {
    const loadingEl = addLoading();
    try {
        const res = await fetch('/start_game');
        const data = await res.json(); // {story, mechanics, choices}
        loadingEl.remove();

        // 1. Hiển thị nội dung mở đầu
        addMessage('dm', data.story);
        // 3. Render 4 nút lựa chọn
        renderChoices(data.choices);

        renderEntities(data.active_entities);
        if (data.act_index !== undefined) updateActLabel(data.act_index);

        if (data.mechanics && (data.mechanics.is_dead || data.mechanics.character_died)) {
            showGameOver();
        }

    } catch(e) {
        console.error('Lỗi khi mở màn:', e);
        loadingEl.remove();
        addMessage('dm', '⚠️ Không thể kết nối tới Người Kể Chuyện.');
    }
}

function showGameOver() {
  isGameOver = true;
  const chatContainer = byId('chat-container');

  const banner = document.createElement('div');
  banner.className = 'death-banner';
  banner.innerHTML = `
    <div class="death-title">☠ ĐÃ CHẾT ☠</div>
    <div class="death-sub">Số phận của ngươi đã được định đoạt.</div>
    <button class="restart-btn" onclick="location.href='/'">Bắt đầu lại</button>
  `;
  chatContainer.appendChild(banner);
  setTimeout(() => { chatContainer.scrollTop = chatContainer.scrollHeight; }, 100);

  // Khóa input vĩnh viễn
  byId('custom-action').disabled = true;
  byId('send-btn').disabled = true;
  byId('custom-action').placeholder = 'Cuộc phiêu lưu đã kết thúc...';
  byId('choices-container').innerHTML = '';
}

function setInputsEnabled(enabled){
  if (isGameOver) return;  // không cho mở lại input nếu đã chết
  byId('custom-action').disabled = !enabled;
  byId('send-btn').disabled = !enabled;
  byId('choices-container').querySelectorAll('.choice-btn').forEach(b => b.disabled = !enabled);
}

function renderDmTurn(data) {
  const container = byId('chat-container');
  const div = document.createElement('div');
  div.className = 'msg dm';

  let html = '';

  const mechanics = data.mechanics || {};

  // 1. Nhật ký thay đổi (đầu khung) — giờ có thêm dòng kết quả roll
  const logParts = [];

  // 1a. Kết quả roll (thành công/thất bại + chi tiết dice nếu có)
  if (mechanics.dice) {
    const { rolls, taken, modifier, total } = mechanics.dice;
    const dc = mechanics.dc;
    const rollTypeLabel = {
      advantage: 'Lợi thế',
      disadvantage: 'Bất lợi',
      normal: 'Thường'
    }[mechanics.roll_type] || '';

    const resultIcon = mechanics.success ? '✅' : '❌';
    const resultLabel = mechanics.success ? 'Thành công' : 'Thất bại';
    const rollDetail = rolls && rolls.length > 1
      ? `[${rolls.join(', ')}] → lấy ${taken}`
      : `${taken}`;

    logParts.push(
      `<div class="turn-roll ${mechanics.success ? 'roll-success' : 'roll-fail'}">`
      + `${resultIcon} <strong>${resultLabel}</strong>`
      + (rollTypeLabel ? ` · ${rollTypeLabel}` : '')
      + ` · Xúc xắc: ${rollDetail} + ${modifier} = <strong>${total}</strong>`
      + (dc !== undefined ? ` (Cần: ${dc})` : '')
      + `</div>`
    );
  }
  // Không có dice (auto success/fail) -> ẩn dòng kết quả, không có gì để "roll" cả.

  // 1b. Thay đổi chỉ số
  const changes = mechanics.changes;
  if (changes) {
    const entries = [];
    if (changes.hp)    entries.push(`❤️ HP ${changes.hp > 0 ? '+' : ''}${changes.hp}`);
    if (changes.mana)  entries.push(`🔷 Mana ${changes.mana > 0 ? '+' : ''}${changes.mana}`);
    if (changes.gold)  entries.push(`💰 Vàng ${changes.gold > 0 ? '+' : ''}${changes.gold}`);
    if (changes.xp)    entries.push(`⭐ XP ${changes.xp > 0 ? '+' : ''}${changes.xp}`);
    // Lọc tên rỗng/null trước khi render — tránh in "🎒 Nhận: null" / "🖐️ Sử
    // dụng: null" khi backend lỡ đưa giá trị null/"null" vào mảng.
    (changes.items_added || []).filter(isRealName).forEach(name => entries.push(`🎒 Nhận: ${escapeHtml(name)}`));
    (changes.items_removed || []).filter(isRealName).forEach(name => entries.push(`🖐️ Sử dụng: ${escapeHtml(name)}`));

    if (entries.length > 0) {
      logParts.push(`<div class="turn-changes">${entries.map(e => `<span class="log-entry">${e}</span>`).join('')}</div>`);
    }
  }

  // 1c. Lên cấp (bug: XP vượt mốc nhưng nhân vật không lên cấp — nay backend
  // xử lý & báo qua mechanics.level_up)
  if (mechanics.level_up && mechanics.level_up.new_level) {
    const lv = mechanics.level_up.new_level;
    logParts.push(`<div class="turn-levelup">⬆️ <strong>Lên cấp ${lv}!</strong> Máu & mana đã hồi đầy.</div>`);
  }

  if (logParts.length > 0) {
    html += `<div class="turn-log">${logParts.join('')}</div>`;
  }

  // 3. Story
  html += `<div class="turn-story">${escapeHtml(data.story)}</div>`;

  // 2. Reasoning
  if (mechanics.reasoning) {
    html += `<div class="turn-reasoning">⚠️ ${escapeHtml(mechanics.reasoning)}</div>`;
  }

  // 4. Nút thử lại — chỉ lượt GẦN NHẤT mới thử lại được (backend chỉ giữ 1
  // snapshot), nút cũ trên bubble trước đó bị gỡ ở nơi gọi hàm này.
  html += `<button type="button" class="retry-btn" title="Thử lại lượt này (nếu model sinh lỗi)" onclick="retryLastTurn()">↻</button>`;

  div.innerHTML = html;
  container.appendChild(div);
  setTimeout(() => { container.scrollTop = container.scrollHeight; }, 100);
  return div;
}

/* ---------------------------- Trợ lý ngoài-truyện ---------------------------- */

let assistantAsking = false;

function addAssistantMessage(kind, text){
  const container = byId('assistant-messages');
  const div = document.createElement('div');
  div.className = 'a-msg ' + kind;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function toggleAssistantPanel(forceOpen){
  const panel = byId('assistant-panel');
  const open = forceOpen !== undefined ? forceOpen : !panel.classList.contains('open');
  panel.classList.toggle('open', open);
  if (open) byId('assistant-input').focus();
}

async function askAssistant(){
  const input = byId('assistant-input');
  const question = input.value.trim();
  if (!question || assistantAsking) return;
  input.value = '';
  assistantAsking = true;
  byId('assistant-send-btn').disabled = true;

  addAssistantMessage('user', question);
  const loadingEl = addAssistantMessage('loading', 'Đang tra cứu…');

  try {
    const res = await fetch('/assistant_ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    loadingEl.remove();
    addAssistantMessage('answer', data.error || data.answer || 'Không có phản hồi.');
  } catch (e) {
    console.error('Lỗi khi hỏi trợ lý:', e);
    loadingEl.remove();
    addAssistantMessage('answer', '⚠️ Không thể kết nối tới trợ lý.');
  } finally {
    assistantAsking = false;
    byId('assistant-send-btn').disabled = false;
  }
}

/* ---------------------------- Dropdown "Chơi lại / Xoá campaign" ---------------------------- */

function togglePanelMenu(force){
  const dd = byId('panel-menu-dropdown');
  if (!dd) return;
  if (force === undefined) dd.classList.toggle('hidden');
  else dd.classList.toggle('hidden', !force);
}

function initPanelMenu(){
  const btn = byId('panel-menu-btn');
  if (!btn) return;
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    togglePanelMenu();
  });
  document.addEventListener('click', (e) => {
    const menu = byId('panel-menu');
    if (!menu || menu.contains(e.target)) return;
    togglePanelMenu(false);
  });
}

let replayPollTimer = null;

function openReplayModal(){
  byId('replay-modal-overlay').classList.remove('hidden');
  byId('replay-error-text').classList.add('hidden');
  byId('replay-progress-fill').style.width = '0%';
  byId('replay-progress-label').textContent = 'Đang chuẩn bị…';
}

const REPLAY_STAGE_LABEL = {
  milestone: 'Đang chuẩn bị chặng đầu…',
  opening: 'Đang mở màn…',
  ready: 'Đã sẵn sàng!',
};

async function pollReplayStatus(){
  if (replayPollTimer){ clearTimeout(replayPollTimer); replayPollTimer = null; }
  let data;
  try{
    const res = await fetch('/setup_status');
    data = await res.json();
  } catch(e){
    console.error('Lỗi khi kiểm tra tiến trình:', e);
    replayPollTimer = setTimeout(pollReplayStatus, 1500);
    return;
  }

  if (data.stage === 'error'){
    byId('replay-error-text').textContent = '⚠️ ' + (data.error || 'Đã có lỗi khi dựng lại thế giới.');
    byId('replay-error-text').classList.remove('hidden');
    return;
  }

  const p = Math.max(0, Math.min(100, Number(data.percent) || 0));
  byId('replay-progress-fill').style.width = p + '%';
  byId('replay-progress-label').textContent = `${REPLAY_STAGE_LABEL[data.stage] || 'Đang xử lý…'} (${p}%)`;

  if (data.stage === 'ready'){
    window.location.reload();
    return;
  }
  replayPollTimer = setTimeout(pollReplayStatus, 1500);
}

async function handleReplayCampaign(mode){
  const label = mode === 'same_character' ? 'chơi lại campaign này' : 'chơi lại campaign với nhân vật mới';
  if (!confirm(`Bạn chắc chắn muốn ${label}? Tiến trình hiện tại sẽ mất.`)) return;
  togglePanelMenu(false);

  let data;
  try{
    const res = await fetch('/replay_campaign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    data = await res.json();
  } catch(e){
    console.error('Lỗi replay_campaign:', e);
    alert('Không kết nối được server.');
    return;
  }

  if (data.error){ alert(data.error); return; }
  if (data.redirect){ window.location.href = data.redirect; return; }

  openReplayModal();
  pollReplayStatus();
}

async function handleDeleteCampaign(){
  if (!confirm('Xoá TOÀN BỘ campaign hiện tại (nhân vật, tiến trình, cốt truyện)? Không thể hoàn tác.')) return;
  togglePanelMenu(false);

  try{
    const res = await fetch('/delete_campaign', { method: 'POST' });
    const data = await res.json();
    window.location.href = data.redirect || '/';
  } catch(e){
    console.error('Lỗi delete_campaign:', e);
    alert('Không kết nối được server.');
  }
}

function initAssistant(){
  byId('assistant-toggle').addEventListener('click', (e) => {
    e.stopPropagation();
    toggleAssistantPanel();
  });
  byId('assistant-close').addEventListener('click', () => toggleAssistantPanel(false));
  byId('assistant-send-btn').addEventListener('click', askAssistant);
  byId('assistant-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') askAssistant();
  });

  // Click ra ngoài panel (và ngoài nút toggle) -> tự đóng.
  document.addEventListener('click', (e) => {
    const panel = byId('assistant-panel');
    if (!panel.classList.contains('open')) return;
    if (panel.contains(e.target)) return;
    toggleAssistantPanel(false);
  });
}

async function init(){
  const char = await loadCharacter();

  byId('custom-action').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendCustomAction();
  });
  initAssistant();
  initPanelMenu();

  if (char){
    startStory();
    fetchAndRenderContext();
  } else {
    addMessage('dm', 'Chưa có nhân vật nào. Hãy quay lại màn hình tạo nhân vật trước.');
  }
}

document.addEventListener('DOMContentLoaded', init);