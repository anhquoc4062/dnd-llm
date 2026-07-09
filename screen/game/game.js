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

let isGameOver = false;

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
  byId('avatar-race').textContent = findIcon(typeof RACES !== 'undefined' ? RACES : null, char.race, '🧑');
  byId('avatar-class').textContent = findIcon(typeof CLASSES !== 'undefined' ? CLASSES : null, char.character_class, '🛡️');
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
    const val = attrs[k] ?? 8;
    el.textContent = val;
    el.classList.remove('pos', 'neg', 'neutral');
    if (val > 8) el.classList.add('pos');
    else if (val < 8) el.classList.add('neg');
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

async function handlePlayerAction(text) {
    if (!text || !text.trim()) return;
    setInputsEnabled(false);
    byId('choices-container').innerHTML = '';
    addMessage('user', text);
    const loadingEl = addLoading();

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });
        const data = await res.json(); // {story, mechanics, choices}
        loadingEl.remove();

        // 1. Hiển thị nội dung truyện
        renderDmTurn(data);

        // 3. Hiển thị lựa chọn
        renderChoices(data.choices);

        if (data.mechanics && (data.mechanics.is_dead || data.mechanics.character_died)) {
            showGameOver();
        }

        // 4. Làm mới chỉ số nhân vật
        loadCharacter();
    } catch (e) {
        console.error('Lỗi:', e);
        loadingEl.remove();
        addMessage('dm', '⚠️ Hệ thống gặp lỗi xử lý.');
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
    (changes.items_added || []).forEach(name => entries.push(`🎒 Nhận: ${escapeHtml(name)}`));
    (changes.items_removed || []).forEach(name => entries.push(`🖐️ Sử dụng: ${escapeHtml(name)}`));

    if (entries.length > 0) {
      logParts.push(`<div class="turn-changes">${entries.map(e => `<span class="log-entry">${e}</span>`).join('')}</div>`);
    }
  }

  if (logParts.length > 0) {
    html += `<div class="turn-log">${logParts.join('')}</div>`;
  }

  // 2. Reasoning
  if (mechanics.reasoning) {
    html += `<div class="turn-reasoning">⚠️ ${escapeHtml(mechanics.reasoning)}</div>`;
  }

  // 3. Story
  html += `<div class="turn-story">${escapeHtml(data.story)}</div>`;

  div.innerHTML = html;
  container.appendChild(div);
  setTimeout(() => { container.scrollTop = container.scrollHeight; }, 100);
  return div;
}

async function init(){
  const char = await loadCharacter();

  byId('custom-action').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendCustomAction();
  });

  if (char){
    startStory();
  } else {
    addMessage('dm', 'Chưa có nhân vật nào. Hãy quay lại màn hình tạo nhân vật trước.');
  }
}

document.addEventListener('DOMContentLoaded', init);