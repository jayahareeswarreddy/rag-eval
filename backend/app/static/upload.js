/* Upload scope: when a session is active, Search and Ask query the visitor's
   own files instead of the demo corpus.

   Depends on app.js for $, $$, api(), API and loadStatus(). */

let sessionId = null;
try {
  sessionId = sessionStorage.getItem('rag-session') || null;
} catch {
  /* storage blocked; the session simply will not survive a reload */
}

function setSession(id) {
  sessionId = id;
  try {
    if (id) sessionStorage.setItem('rag-session', id);
    else sessionStorage.removeItem('rag-session');
  } catch {
    /* non-fatal */
  }
}

/** Current scope, read by app.js before every search and ask. */
window.ragScope = () => sessionId;

const DEMO_EXAMPLES = {
  q: [
    'how do I shuffle a list randomly',
    'create a temporary directory',
    'wrap text to a fixed width',
    'encode binary data as ascii text',
  ],
  aq: [
    'how are temporary files created safely',
    'what does a thread lock do',
    'how is a zip archive written',
  ],
};

/** Prompts drawn from the visitor's own filenames, so they are answerable. */
function uploadExamples(names) {
  const stems = names.map((n) => n.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ').trim());
  return {
    q: stems.slice(0, 4).map((s) => `${s} rules`),
    aq: stems.slice(0, 3).map((s) => `summarise ${s}`),
  };
}

function setExamples(sets) {
  Object.entries(sets).forEach(([target, prompts]) => {
    const row = document.querySelector(`.ex[data-target="${target}"]`)?.parentElement;
    if (!row) return;
    row.querySelectorAll('.ex').forEach((el) => el.remove());
    prompts.forEach((text) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'ex';
      b.dataset.target = target;
      b.textContent = text;
      b.addEventListener('click', () => {
        const input = $('#' + target);
        input.value = text;
        input.focus();
        input.form.requestSubmit();
      });
      row.appendChild(b);
    });
  });
}

function renderScope(info) {
  const state = $('#scope-state');
  window.ragScopeInfo = info || null;
  if (!info) {
    state.hidden = true;
    setExamples(DEMO_EXAMPLES);
    return;
  }
  state.hidden = false;
  setExamples(uploadExamples(info.filenames || []));

  const names = info.filenames || [];
  const shown = names.slice(0, 4).join(', ');
  const passages = info.chunks?.count ?? 0;
  $('#scope-files').textContent =
    `${names.length} file${names.length === 1 ? '' : 's'} · ${passages} passages — ` +
    `${shown}${names.length > 4 ? ', …' : ''}`;

  const skipped = info.skipped || [];
  $('#scope-skipped').textContent = skipped.length
    ? `Skipped: ${skipped.map((s) => `${s.file} (${s.reason})`).join('; ')}`
    : '';
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;

  $('#dropzone').classList.remove('is-over');
  $('#scope-state').hidden = false;
  $('#scope-files').textContent =
    `uploading ${files.length} file${files.length === 1 ? '' : 's'}…`;
  $('#scope-skipped').textContent = '';

  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  const url = sessionId
    ? `/upload?session_id=${encodeURIComponent(sessionId)}`
    : '/upload';

  try {
    // Deliberately not using api(): this is multipart, so the JSON
    // Content-Type header must be omitted for the browser to set the
    // multipart boundary itself.
    const res = await fetch(API + url, { method: 'POST', body: form });
    const body = await res.json().catch(() => ({}));

    if (!res.ok) {
      const detail = body.detail;
      const reason =
        typeof detail === 'string'
          ? detail
          : detail?.skipped?.length
            ? detail.skipped.map((s) => `${s.file}: ${s.reason}`).join('; ')
            : `Upload failed (HTTP ${res.status})`;
      $('#scope-files').textContent = '';
      $('#scope-skipped').textContent = reason;
      return;
    }

    setSession(body.session_id);
    renderScope(body);
    loadStatus();
  } catch (err) {
    $('#scope-files').textContent = '';
    $('#scope-skipped').textContent = err.message;
  }
}

$('#browse').addEventListener('click', () => $('#file-input').click());

$('#file-input').addEventListener('change', (e) => {
  uploadFiles(e.target.files);
  e.target.value = ''; // so picking the same file again still fires change
});

const dropzone = $('#dropzone');

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-over');
  })
);

dropzone.addEventListener('dragleave', (e) => {
  // Moving between children fires dragleave; only clear on a real exit.
  if (dropzone.contains(e.relatedTarget)) return;
  dropzone.classList.remove('is-over');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('is-over');
  uploadFiles(e.dataTransfer.files);
});

// Without this, dropping outside the zone makes the browser navigate away
// to the file instead of staying on the page.
['dragover', 'drop'].forEach((evt) =>
  document.addEventListener(evt, (e) => {
    if (!dropzone.contains(e.target)) e.preventDefault();
  })
);

$('#clear-docs').addEventListener('click', async () => {
  const id = sessionId;
  setSession(null);
  renderScope(null);
  loadStatus();
  if (!id) return;
  try {
    await fetch(`${API}/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
  } catch {
    /* the TTL collects it regardless */
  }
});

/** Restore a session after a reload, dropping it if the server expired it. */
async function restoreSession() {
  if (!sessionId) return;
  try {
    renderScope(await api(`/sessions/${encodeURIComponent(sessionId)}`));
  } catch {
    setSession(null);
    renderScope(null);
  }
}

/* --------------------------- example chips ---------------------------- */

$$('.ex').forEach((chip) =>
  chip.addEventListener('click', () => {
    const input = $('#' + chip.dataset.target);
    input.value = chip.textContent.trim();
    input.focus();
    input.form.requestSubmit();
  })
);

/* ------------------------------- intro -------------------------------- */

try {
  if (localStorage.getItem('rag-intro-hidden') === '1') $('#intro').hidden = true;
} catch {
  /* non-fatal */
}

$('#intro-close').addEventListener('click', () => {
  $('#intro').hidden = true;
  try {
    localStorage.setItem('rag-intro-hidden', '1');
  } catch {
    /* non-fatal */
  }
});

/* --------------------------- limits + boot ---------------------------- */

async function loadLimits() {
  try {
    const { limits } = await api('/upload-limits');
    const kb = Math.round(limits.max_file_bytes / 1000);
    $('#dz-limits').textContent =
      `Text and source files · up to ${limits.max_files} files, ${kb}KB each · ` +
      'held in memory for one hour, never written to disk';
  } catch {
    /* the static fallback text in the markup stands */
  }
}

loadLimits();
restoreSession();
