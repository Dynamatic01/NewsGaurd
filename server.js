// ═══════════════════════════════════════════════════════════════════════════════
// NewsGuard — Full Analysis Pipeline Server
// Pipeline: Input → Content Extraction → AI Model → Source Trust → Fact-Check → Result
// ═══════════════════════════════════════════════════════════════════════════════
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const cheerio = require('cheerio');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: '*', methods: ['GET','POST'], allowedHeaders: ['Content-Type'] }));
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname)));

// ─── API Keys ────────────────────────────────────────────────────────────────
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const GEMINI_KEY    = process.env.GEMINI_API_KEY;
const NEWSAPI_KEY   = process.env.NEWSAPI_KEY;
const GOOGLE_FC_KEY = process.env.GOOGLE_FACTCHECK_KEY;
const GNEWS_KEY     = process.env.GNEWS_KEY;
function hasKey(k) { return k && k.length > 5 && !k.includes('your_'); }

// ─── Local ML Server ─────────────────────────────────────────────────────────
const ML_SERVER = 'http://127.0.0.1:8000';
let mlServerOnline = false;   // cached after first health-check

async function checkMLServer() {
  try {
    const r = await fetchWithTimeout(`${ML_SERVER}/api/ml/health`, {}, 3000);
    if (r.ok) {
      const d = await r.json();
      mlServerOnline = d.classical_model === true;
      if (mlServerOnline) console.log(`  🤖 Local ML Server: ✅ online  (${d.model_name}, acc=${(d.model_accuracy*100).toFixed(1)}%)`);
      else console.log('  🤖 Local ML Server: ⚠️  online but model not loaded — run train_model.py');
    }
  } catch {
    mlServerOnline = false;
  }
  return mlServerOnline;
}

async function callLocalML(text) {
  if (!text || text.length < 15) return null;
  try {
    const res = await fetchWithTimeout(`${ML_SERVER}/api/ml/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.slice(0, 5000) })
    }, 5000);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.error) return null;
    return data;    // { result, trust_score, confidence, ml_prob, signals }
  } catch {
    return null;
  }
}

async function callSimilarityCheck(text, title = '') {
  if (!text || text.length < 20) return null;
  try {
    const res = await fetchWithTimeout(`${ML_SERVER}/api/similarity/check`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.slice(0, 4000), title: title.slice(0, 200) })
    }, 7000);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.verdict === 'UNAVAILABLE') return null;
    return data;  // { verdict, max_score, top_matches, explanation, ... }
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE STEP 1: CONTENT EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════════
async function extractContent(url) {
  const result = { status: 'pending', title: '', text: '', meta: {}, wordCount: 0 };
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; NewsGuard/1.0; +https://newsguard.ai)',
        'Accept': 'text/html,application/xhtml+xml'
      }
    });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const html = await res.text();
    const $ = cheerio.load(html);

    // Remove noise
    $('script, style, nav, footer, header, aside, iframe, noscript, .ad, .advertisement, .sidebar').remove();

    // Extract metadata
    result.title = $('meta[property="og:title"]').attr('content')
      || $('title').text()
      || $('h1').first().text()
      || '';
    result.meta.description = $('meta[property="og:description"]').attr('content')
      || $('meta[name="description"]').attr('content') || '';
    result.meta.siteName = $('meta[property="og:site_name"]').attr('content') || '';
    result.meta.author = $('meta[name="author"]').attr('content')
      || $('[rel="author"]').text()
      || $('[class*="author"]').first().text() || '';
    result.meta.publishDate = $('meta[property="article:published_time"]').attr('content')
      || $('time').attr('datetime') || '';

    // Extract article text
    const articleSelectors = ['article', '[role="main"]', '.article-body', '.story-body',
      '.post-content', '.entry-content', '.article-content', 'main'];
    let articleText = '';
    for (const sel of articleSelectors) {
      const el = $(sel);
      if (el.length && el.text().trim().length > 100) {
        articleText = el.find('p').map((_, p) => $(p).text().trim()).get().join('\n\n');
        break;
      }
    }
    if (!articleText) {
      articleText = $('p').map((_, p) => $(p).text().trim()).get()
        .filter(t => t.length > 30).join('\n\n');
    }

    result.text = articleText.slice(0, 4000);
    result.wordCount = articleText.split(/\s+/).length;
    result.status = result.text.length > 50 ? 'success' : 'partial';
  } catch (err) {
    result.status = 'failed';
    result.error = err.name === 'AbortError' ? 'Timeout fetching URL' : err.message;
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE STEP 2: SOURCE TRUST DATABASE & RELIABILITY SCORING ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

// ── SENSATIONAL WORDS (trigger penalty) ──────────────────────────────────────
const SENSATIONAL_WORDS = [
  'shocking','bombshell','exposed','banned','miracle','secret','hidden',
  'deep state','hoax','conspiracy','leaked','disturbing','horrifying',
  'unbelievable','mind-blowing','you won\'t believe','goes viral',
  'explosive','scandalous','cover-up','suppressed','censored',
  'urgent warning','wake up','truth revealed','doctors don\'t want',
  'government hiding','big pharma','false flag','globalist',
  'they don\'t want you','mainstream media is lying'
];

// ── TRUSTED DOMAIN DATABASE (120+ sources) ───────────────────────────────────
const SOURCE_TRUST_DB = {
  // ════ TIER 1 — HIGHLY TRUSTED (85-100) ════
  // Wire Services
  'reuters.com':    { score:97, tier:1, category:'Wire Service',       bias:'Center',       flag:'🌐', notes:'Global gold-standard wire service' },
  'apnews.com':     { score:96, tier:1, category:'Wire Service',       bias:'Center',       flag:'🌐', notes:'Associated Press — fact-first reporting' },
  'afp.com':        { score:94, tier:1, category:'Wire Service',       bias:'Center',       flag:'🌐', notes:'Agence France-Presse' },
  'pti.in':         { score:90, tier:1, category:'Wire Service',       bias:'Center',       flag:'🇮🇳', notes:'Press Trust of India' },
  'ani.in':         { score:85, tier:1, category:'Wire Service',       bias:'Center',       flag:'🇮🇳', notes:'Asian News International' },
  // Public Broadcasters
  'bbc.com':        { score:93, tier:1, category:'Public Broadcaster', bias:'Center-Left',  flag:'🇬🇧', notes:'UK public broadcaster, strong editorial standards' },
  'bbc.co.uk':      { score:93, tier:1, category:'Public Broadcaster', bias:'Center-Left',  flag:'🇬🇧', notes:'UK public broadcaster' },
  'npr.org':        { score:91, tier:1, category:'Public Broadcaster', bias:'Center-Left',  flag:'🇺🇸', notes:'US National Public Radio' },
  'pbs.org':        { score:90, tier:1, category:'Public Broadcaster', bias:'Center',       flag:'🇺🇸', notes:'US Public Broadcasting Service' },
  'abc.net.au':     { score:88, tier:1, category:'Public Broadcaster', bias:'Center-Left',  flag:'🇦🇺', notes:'Australian Broadcasting Corporation' },
  'dw.com':         { score:89, tier:1, category:'Public Broadcaster', bias:'Center',       flag:'🇩🇪', notes:'Deutsche Welle — German international broadcaster' },
  // Scientific / Medical
  'nature.com':     { score:98, tier:1, category:'Scientific Journal', bias:'Center',       flag:'🔬', notes:'Peer-reviewed science journal' },
  'science.org':    { score:97, tier:1, category:'Scientific Journal', bias:'Center',       flag:'🔬', notes:'AAAS Science journal' },
  'thelancet.com':  { score:96, tier:1, category:'Medical Journal',    bias:'Center',       flag:'🏥', notes:'Top medical journal' },
  'nejm.org':       { score:96, tier:1, category:'Medical Journal',    bias:'Center',       flag:'🏥', notes:'New England Journal of Medicine' },
  'bmj.com':        { score:95, tier:1, category:'Medical Journal',    bias:'Center',       flag:'🏥', notes:'British Medical Journal' },
  'pubmed.ncbi.nlm.nih.gov': { score:97, tier:1, category:'Medical DB', bias:'Center',     flag:'🔬', notes:'US National Library of Medicine' },
  // International Organizations
  'who.int':        { score:95, tier:1, category:'Intl Organization',  bias:'Center',       flag:'🌍', notes:'World Health Organization' },
  'un.org':         { score:92, tier:1, category:'Intl Organization',  bias:'Center',       flag:'🌍', notes:'United Nations' },
  'worldbank.org':  { score:91, tier:1, category:'Intl Organization',  bias:'Center',       flag:'🌍', notes:'World Bank' },
  'imf.org':        { score:90, tier:1, category:'Intl Organization',  bias:'Center',       flag:'🌍', notes:'International Monetary Fund' },
  'unicef.org':     { score:93, tier:1, category:'Intl Organization',  bias:'Center',       flag:'🌍', notes:'UNICEF' },
  // Government (Official)
  'nih.gov':        { score:94, tier:1, category:'Government Science', bias:'Center',       flag:'🏛️', notes:'US National Institutes of Health' },
  'cdc.gov':        { score:93, tier:1, category:'Government Health',  bias:'Center',       flag:'🏛️', notes:'US Centers for Disease Control' },
  'fda.gov':        { score:91, tier:1, category:'Government',         bias:'Center',       flag:'🏛️', notes:'US Food & Drug Administration' },
  'nasa.gov':       { score:95, tier:1, category:'Government Science', bias:'Center',       flag:'🏛️', notes:'NASA — space & science authority' },
  'gov.uk':         { score:90, tier:1, category:'Government',         bias:'Center',       flag:'🇬🇧', notes:'UK Government' },
  'pib.gov.in':     { score:88, tier:1, category:'Government',         bias:'Center',       flag:'🇮🇳', notes:'Press Information Bureau India' },
  'india.gov.in':   { score:87, tier:1, category:'Government',         bias:'Center',       flag:'🇮🇳', notes:'Indian Government portal' },
  'mea.gov.in':     { score:86, tier:1, category:'Government',         bias:'Center',       flag:'🇮🇳', notes:'Ministry of External Affairs, India' },
  // Fact-Checkers
  'snopes.com':     { score:90, tier:1, category:'Fact-Checker',       bias:'Center',       flag:'✅', notes:'Established independent fact-checker' },
  'factcheck.org':  { score:91, tier:1, category:'Fact-Checker',       bias:'Center',       flag:'✅', notes:'Annenberg Public Policy Center' },
  'politifact.com': { score:89, tier:1, category:'Fact-Checker',       bias:'Center',       flag:'✅', notes:'Pulitzer Prize-winning fact-checker' },
  'fullfact.org':   { score:90, tier:1, category:'Fact-Checker',       bias:'Center',       flag:'✅', notes:'UK independent fact-checker' },
  'boomlive.in':    { score:86, tier:1, category:'Fact-Checker',       bias:'Center',       flag:'🇮🇳', notes:'Indian digital fact-checking outlet' },
  'altnews.in':     { score:85, tier:1, category:'Fact-Checker',       bias:'Center-Left',  flag:'🇮🇳', notes:'Indian fact-checking site' },
  'vishvasnews.com':{ score:84, tier:1, category:'Fact-Checker',       bias:'Center',       flag:'🇮🇳', notes:'India Today fact-check partner' },
  // ════ TIER 2 — GENERALLY TRUSTED (65-84) ════
  'nytimes.com':    { score:82, tier:2, category:'Newspaper',          bias:'Center-Left',  flag:'🇺🇸', notes:'US paper of record' },
  'washingtonpost.com': { score:80, tier:2, category:'Newspaper',      bias:'Center-Left',  flag:'🇺🇸', notes:'US national newspaper' },
  'wsj.com':        { score:81, tier:2, category:'Newspaper',          bias:'Center-Right', flag:'🇺🇸', notes:'Wall Street Journal' },
  'theguardian.com':{ score:79, tier:2, category:'Newspaper',          bias:'Center-Left',  flag:'🇬🇧', notes:'UK left-leaning newspaper' },
  'economist.com':  { score:84, tier:2, category:'Magazine',           bias:'Center',       flag:'🇬🇧', notes:'UK economics & world affairs' },
  'ft.com':         { score:83, tier:2, category:'Newspaper',          bias:'Center',       flag:'🇬🇧', notes:'Financial Times' },
  'time.com':       { score:76, tier:2, category:'Magazine',           bias:'Center-Left',  flag:'🇺🇸', notes:'Time magazine' },
  'newsweek.com':   { score:68, tier:2, category:'Magazine',           bias:'Center-Left',  flag:'🇺🇸', notes:'Newsweek' },
  'forbes.com':     { score:74, tier:2, category:'Business Media',     bias:'Center-Right', flag:'🇺🇸', notes:'Forbes business news' },
  'bloomberg.com':  { score:80, tier:2, category:'Business Media',     bias:'Center',       flag:'🇺🇸', notes:'Bloomberg financial news' },
  'thehindu.com':   { score:78, tier:2, category:'Newspaper',          bias:'Center-Left',  flag:'🇮🇳', notes:'Indian national newspaper' },
  'indianexpress.com': { score:76, tier:2, category:'Newspaper',       bias:'Center',       flag:'🇮🇳', notes:'Indian Express newspaper' },
  'hindustantimes.com': { score:72, tier:2, category:'Newspaper',      bias:'Center',       flag:'🇮🇳', notes:'Hindustan Times' },
  'timesofindia.indiatimes.com': { score:70, tier:2, category:'Newspaper', bias:'Center',   flag:'🇮🇳', notes:'Times of India' },
  'livemint.com':   { score:75, tier:2, category:'Newspaper',          bias:'Center',       flag:'🇮🇳', notes:'Mint — Indian financial daily' },
  'business-standard.com': { score:74, tier:2, category:'Newspaper',   bias:'Center',       flag:'🇮🇳', notes:'Business Standard India' },
  'telegraphindia.com': { score:72, tier:2, category:'Newspaper',      bias:'Center-Left',  flag:'🇮🇳', notes:'The Telegraph India' },
  'deccanherald.com': { score:70, tier:2, category:'Newspaper',        bias:'Center-Left',  flag:'🇮🇳', notes:'Deccan Herald' },
  'ndtv.com':       { score:73, tier:2, category:'News Channel',       bias:'Center-Left',  flag:'🇮🇳', notes:'NDTV news channel' },
  'indiatoday.in':  { score:72, tier:2, category:'News Channel',       bias:'Center',       flag:'🇮🇳', notes:'India Today media group' },
  'firstpost.com':  { score:68, tier:2, category:'Online Media',       bias:'Center-Right', flag:'🇮🇳', notes:'Firstpost digital news' },
  'scroll.in':      { score:72, tier:2, category:'Online Media',       bias:'Center-Left',  flag:'🇮🇳', notes:'Scroll.in digital journalism' },
  'theprint.in':    { score:73, tier:2, category:'Online Media',       bias:'Center',       flag:'🇮🇳', notes:'The Print — India analysis' },
  'thequint.com':   { score:70, tier:2, category:'Online Media',       bias:'Center-Left',  flag:'🇮🇳', notes:'The Quint digital media' },
  'newslaundry.com':{ score:74, tier:2, category:'Online Media',       bias:'Center-Left',  flag:'🇮🇳', notes:'Newslaundry media criticism' },
  'aljazeera.com':  { score:72, tier:2, category:'News Channel',       bias:'Center-Left',  flag:'🌐', notes:'Qatar-based international news' },
  'cnn.com':        { score:70, tier:2, category:'Cable News',         bias:'Left',         flag:'🇺🇸', notes:'US cable news network' },
  'abcnews.go.com': { score:74, tier:2, category:'Broadcast News',     bias:'Center-Left',  flag:'🇺🇸', notes:'ABC News US' },
  'cbsnews.com':    { score:74, tier:2, category:'Broadcast News',     bias:'Center-Left',  flag:'🇺🇸', notes:'CBS News US' },
  'nbcnews.com':    { score:73, tier:2, category:'Broadcast News',     bias:'Center-Left',  flag:'🇺🇸', notes:'NBC News US' },
  'sky.com':        { score:72, tier:2, category:'News Channel',       bias:'Center-Right', flag:'🇬🇧', notes:'Sky News UK' },
  'france24.com':   { score:76, tier:2, category:'News Channel',       bias:'Center',       flag:'🇫🇷', notes:'France 24 international' },
  'channelnewsasia.com': { score:74, tier:2, category:'News Channel',  bias:'Center',       flag:'🇸🇬', notes:'CNA — Singapore broadcast' },
  'theverge.com':   { score:72, tier:2, category:'Tech Media',         bias:'Center-Left',  flag:'💻', notes:'The Verge tech journalism' },
  'wired.com':      { score:74, tier:2, category:'Tech Media',         bias:'Center-Left',  flag:'💻', notes:'Wired technology magazine' },
  'techcrunch.com': { score:70, tier:2, category:'Tech Media',         bias:'Center',       flag:'💻', notes:'TechCrunch startup news' },
  'arstechnica.com':{ score:76, tier:2, category:'Tech Media',         bias:'Center-Left',  flag:'💻', notes:'Ars Technica in-depth tech' },
  'mit.edu':        { score:88, tier:2, category:'Academic',           bias:'Center',       flag:'🎓', notes:'MIT — research institution' },
  // ════ TIER 3 — MIXED RELIABILITY (35-64) ════
  'foxnews.com':    { score:55, tier:3, category:'Cable News',         bias:'Right',        flag:'🇺🇸', notes:'Opinion-heavy US cable news' },
  'msnbc.com':      { score:58, tier:3, category:'Cable News',         bias:'Left',         flag:'🇺🇸', notes:'Opinion-heavy US cable news' },
  'huffpost.com':   { score:58, tier:3, category:'Online Media',       bias:'Left',         flag:'🇺🇸', notes:'Opinion-heavy digital outlet' },
  'nypost.com':     { score:50, tier:3, category:'Tabloid',            bias:'Right',        flag:'🇺🇸', notes:'New York Post tabloid' },
  'dailymail.co.uk':{ score:45, tier:3, category:'Tabloid',            bias:'Right',        flag:'🇬🇧', notes:'UK tabloid — sensational reporting' },
  'thesun.co.uk':   { score:40, tier:3, category:'Tabloid',            bias:'Right',        flag:'🇬🇧', notes:'UK tabloid' },
  'buzzfeed.com':   { score:52, tier:3, category:'Online Media',       bias:'Left',         flag:'🇺🇸', notes:'Viral content, mixed quality' },
  'vox.com':        { score:62, tier:3, category:'Online Media',       bias:'Left',         flag:'🇺🇸', notes:'Vox explanatory journalism' },
  'breitbart.com':  { score:35, tier:3, category:'Online Media',       bias:'Far-Right',    flag:'🇺🇸', notes:'Far-right, heavily biased' },
  'thedailywire.com':{ score:40, tier:3, category:'Online Media',      bias:'Right',        flag:'🇺🇸', notes:'Conservative opinion site' },
  'salon.com':      { score:50, tier:3, category:'Online Media',       bias:'Left',         flag:'🇺🇸', notes:'Progressive opinion outlet' },
  'opindia.com':    { score:42, tier:3, category:'Online Media',       bias:'Right',        flag:'🇮🇳', notes:'Indian right-wing opinion' },
  'thewire.in':     { score:60, tier:3, category:'Online Media',       bias:'Left',         flag:'🇮🇳', notes:'Indian left-leaning digital media' },
  'swarajyamag.com':{ score:48, tier:3, category:'Online Media',       bias:'Right',        flag:'🇮🇳', notes:'Indian conservative opinion magazine' },
  'organiser.org':  { score:40, tier:3, category:'Online Media',       bias:'Far-Right',    flag:'🇮🇳', notes:'RSS-affiliated publication' },
  'newsclick.in':   { score:50, tier:3, category:'Online Media',       bias:'Left',         flag:'🇮🇳', notes:'Left-leaning digital media India' },
  'postcard.news':  { score:20, tier:3, category:'Online Media',       bias:'Far-Right',    flag:'🇮🇳', notes:'Known for viral misinformation in India' },
  'theonion.com':   { score:50, tier:3, category:'Satire',             bias:'Center',       flag:'😂', notes:'SATIRE — not real news' },
  'babylonbee.com': { score:50, tier:3, category:'Satire',             bias:'Right',        flag:'😂', notes:'SATIRE — not real news' },
  // ════ TIER 4 — UNRELIABLE / MISINFORMATION (0-34) ════
  'infowars.com':   { score:5,  tier:4, category:'Conspiracy',         bias:'Far-Right',    flag:'🚨', notes:'Known major misinformation source' },
  'naturalnews.com':{ score:8,  tier:4, category:'Pseudoscience',      bias:'Far-Right',    flag:'🚨', notes:'Anti-vax, anti-science misinformation' },
  'beforeitsnews.com': { score:10, tier:4, category:'Conspiracy',      bias:'Far-Right',    flag:'🚨', notes:'User-generated conspiracy content' },
  'zerohedge.com':  { score:22, tier:4, category:'Conspiracy',         bias:'Far-Right',    flag:'🚨', notes:'Financial conspiracy, often inaccurate' },
  'globalresearch.ca': { score:12, tier:4, category:'Conspiracy',      bias:'Far-Left',     flag:'🚨', notes:'Anti-Western disinfo hub' },
  'yournewswire.com': { score:5, tier:4, category:'Hoax Sites',        bias:'Mixed',        flag:'🚨', notes:'Known fake news site' },
  'worldnewsdailyreport.com': { score:5, tier:4, category:'Hoax Sites', bias:'Mixed',       flag:'🚨', notes:'Fabricated news site' },
  'viralstories.in':{ score:10, tier:4, category:'Hoax Sites',         bias:'Mixed',        flag:'🚨', notes:'Indian viral misinformation' },
  'thelastlineofdefense.org': { score:5, tier:4, category:'Satire/Hoax', bias:'Far-Right',  flag:'🚨', notes:'Spread as real news despite satire label' },
};

// ── TIER INFO ─────────────────────────────────────────────────────────────────
const TIER_INFO = {
  1: { label:'Highly Trusted',    color:'#10b981', bg:'rgba(16,185,129,0.15)', emoji:'✅' },
  2: { label:'Generally Trusted', color:'#3b82f6', bg:'rgba(59,130,246,0.15)', emoji:'🔵' },
  3: { label:'Mixed Reliability', color:'#f59e0b', bg:'rgba(245,158,11,0.15)', emoji:'⚠️' },
  4: { label:'Unreliable Source', color:'#ef4444', bg:'rgba(239,68,68,0.15)',  emoji:'🚨' },
  0: { label:'Unknown Source',    color:'#8b5cf6', bg:'rgba(139,92,246,0.15)', emoji:'❓' },
};

// ── DOMAIN LOOKUP (handles subdomains & TLDs) ─────────────────────────────────
function lookupDomain(domain) {
  const clean = domain.replace(/^www\./, '').toLowerCase().trim();
  if (SOURCE_TRUST_DB[clean]) return { ...SOURCE_TRUST_DB[clean], domain: clean, inDb: true };
  const parts = clean.split('.');
  if (parts.length > 2) {
    for (let i = 1; i < parts.length - 1; i++) {
      const sub = parts.slice(i).join('.');
      if (SOURCE_TRUST_DB[sub]) return { ...SOURCE_TRUST_DB[sub], domain: sub, inDb: true };
    }
  }
  if (/\.(gov|gov\.in|gov\.uk|gov\.au|mil)$/i.test(clean))
    return { score:85, tier:1, category:'Government', bias:'Center', flag:'🏛️', notes:'Official government domain', domain:clean, inDb:false };
  if (/\.(edu|ac\.uk|ac\.in)$/i.test(clean))
    return { score:80, tier:2, category:'Academic', bias:'Center', flag:'🎓', notes:'Educational institution', domain:clean, inDb:false };
  return null;
}

// ── CORE SCORING ENGINE ───────────────────────────────────────────────────────
// Scoring Formula:
//   +40  Trusted / known domain (Tier 1)
//   +30  Tier 2 domain
//   +10  Tier 3 domain
//   +10  HTTPS used
//   +10  Author present
//   +10  Publish date present
//   −20  Sensational words detected (up to 2 penalties of −10 each)
function computeSourceReliability({ url='', title='', content='', author='', publishDate='', httpsUsed=null }={}) {
  let domain = '';
  let isHttps = httpsUsed;
  try {
    const parsed = new URL(url.startsWith('http') ? url : 'https://' + url);
    domain = parsed.hostname;
    if (isHttps === null) isHttps = parsed.protocol === 'https:';
  } catch {
    domain = url.replace(/^https?:\/\//, '').split('/')[0];
    if (isHttps === null) isHttps = url.startsWith('https');
  }

  const dbEntry = lookupDomain(domain);
  const breakdown = [];
  let score = 0;

  // Rule 1: Domain trust
  if (dbEntry) {
    const pts = dbEntry.tier === 1 ? 40 : dbEntry.tier === 2 ? 30 : dbEntry.tier === 3 ? 10 : 0;
    score += pts;
    const tierLabel = TIER_INFO[dbEntry.tier]?.label || 'Unknown';
    breakdown.push({ rule:'Domain Reputation', points: pts >= 0 ? `+${pts}` : `${pts}`, positive: pts > 0,
      detail: dbEntry.inDb ? `${dbEntry.flag || ''} In database — Tier ${dbEntry.tier}: ${tierLabel}` : `Auto-detected ${dbEntry.category}` });
  } else {
    breakdown.push({ rule:'Domain Reputation', points:'+0', positive:false, detail:'❓ Domain not in our database — unknown reliability' });
  }

  // Rule 2: HTTPS
  if (isHttps) { score += 10; breakdown.push({ rule:'HTTPS Secure', points:'+10', positive:true, detail:'🔒 Site uses HTTPS encryption' }); }
  else { breakdown.push({ rule:'HTTPS Secure', points:'+0', positive:false, detail:'⚠️ No HTTPS — may be insecure' }); }

  // Rule 3: Author
  const hasAuthor = typeof author === 'string' && author.trim().length > 1;
  if (hasAuthor) { score += 10; breakdown.push({ rule:'Author Present', points:'+10', positive:true, detail:`✍️ Author: "${author.trim()}"` }); }
  else { breakdown.push({ rule:'Author Present', points:'+0', positive:false, detail:'👤 No author information found' }); }

  // Rule 4: Publish date
  const hasDate = typeof publishDate === 'string' && publishDate.trim().length > 3;
  if (hasDate) { score += 10; breakdown.push({ rule:'Publish Date', points:'+10', positive:true, detail:`📅 Dated: ${publishDate.trim().slice(0,20)}` }); }
  else { breakdown.push({ rule:'Publish Date', points:'+0', positive:false, detail:'📅 No publish date found' }); }

  // Rule 5: Sensational words
  const searchText = `${title} ${content}`.toLowerCase();
  const sensationalHits = SENSATIONAL_WORDS.filter(w => searchText.includes(w.toLowerCase()));
  if (sensationalHits.length > 0) {
    const penalty = Math.min(sensationalHits.length, 2) * 10;
    score -= penalty;
    breakdown.push({ rule:'Sensational Language', points:`-${penalty}`, positive:false,
      detail:`🎭 ${sensationalHits.length} trigger word(s): "${sensationalHits.slice(0,3).join('", "')}"` });
  } else {
    breakdown.push({ rule:'Sensational Language', points:'+0', positive:true, detail:'✅ No sensational language detected' });
  }

  const finalScore = Math.max(0, Math.min(100, score));
  // Tier: use score-based calc, but lock Tier 1 DB entries ≥ 30pts to Tier 1 (domain-only check)
  let tier = finalScore >= 70 ? 1 : finalScore >= 50 ? 2 : finalScore >= 30 ? 3 : 4;
  if (dbEntry) {
    // Never downgrade a known Tier 1 domain below Tier 2 on domain-only checks
    if (dbEntry.tier === 1 && tier > 2) tier = 2;
    // Always lock Tier 4 (misinformation) domains
    if (dbEntry.tier === 4) tier = 4;
    // Tier 3 never gets upgraded above Tier 3
    if (dbEntry.tier === 3 && tier < 3) tier = 3;
  }
  const tierInfo = TIER_INFO[tier];
  const verdicts = { 1:'Reliable Source', 2:'Generally Trustworthy', 3:'Exercise Caution', 4:'Unreliable — Do Not Trust' };
  const recs = {
    1:'This source demonstrates strong editorial standards. Content can generally be trusted.',
    2:'This is generally trustworthy. Verify important claims with a second source.',
    3:'This source has mixed reliability. Cross-check all claims before sharing.',
    4:'This source is known to publish misinformation. Avoid sharing without verification.'
  };
  return { domain, finalScore, tier, tierInfo, verdict:verdicts[tier], recommendation:recs[tier],
           breakdown, sensationalHits, domainInfo: dbEntry || { score:0, tier:0, category:'Unknown', bias:'Unknown', notes:'Not in database', domain, inDb:false },
           isHttps, hasAuthor, hasDate };
}

// ─── Legacy wrapper (used by URL pipeline) ───────────────────────────────────
function getSourceTrustScore(domain) {
  const clean = domain.replace(/^www\./, '').toLowerCase();
  const entry = lookupDomain(clean);
  if (entry) return { ...entry, domain: clean, matched: true };
  if (/\.(gov|gov\.in|edu|ac\.uk)$/i.test(clean))
    return { score:80, category:'Government/Education', bias:'Center', notes:'Official domain', domain:clean, matched:true };
  return { score:50, category:'Unknown', bias:'Unknown', notes:'Source not in database — verify independently', domain:clean, matched:false };
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE STEP 3: AI FAKE NEWS MODEL (Anthropic primary, Gemini fallback)
// ═══════════════════════════════════════════════════════════════════════════════
async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timer);
    return res;
  } catch (err) { clearTimeout(timer); throw err; }
}

// ─── Anthropic (Claude) ──────────────────────────────────────────────────────
async function runAnthropicText(prompt) {
  const res = await fetchWithTimeout('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': ANTHROPIC_KEY, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: 'claude-sonnet-4-20250514', max_tokens: 1500, messages: [{ role: 'user', content: prompt }] })
  }, 25000);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || `Anthropic API error ${res.status}`);
  return data.content[0].text.trim().replace(/```json|```/g, '').trim();
}

// ─── Google Gemini (Fallback) ────────────────────────────────────────────────
async function runGeminiText(prompt) {
  const res = await fetchWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { maxOutputTokens: 1500, temperature: 0.3 } })
    }, 25000
  );
  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || `Gemini API error ${res.status}`);
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error('Gemini returned empty response');
  return text.trim().replace(/```json|```/g, '').trim();
}

// ─── Smart AI Runner (tries Anthropic → Gemini fallback) ────────────────────
async function callAI(prompt) {
  // Try Anthropic first
  if (hasKey(ANTHROPIC_KEY)) {
    try {
      console.log('  → Trying Anthropic Claude...');
      const result = await runAnthropicText(prompt);
      console.log('  ✅ Anthropic succeeded');
      return { text: result, engine: 'anthropic' };
    } catch (err) {
      console.warn('  ⚠️ Anthropic failed:', err.message);
    }
  }
  // Fallback to Gemini
  if (hasKey(GEMINI_KEY)) {
    try {
      console.log('  → Falling back to Google Gemini...');
      const result = await runGeminiText(prompt);
      console.log('  ✅ Gemini succeeded');
      return { text: result, engine: 'gemini' };
    } catch (err) {
      console.warn('  ❌ Gemini also failed:', err.message);
      throw new Error('Gemini fallback failed: ' + err.message);
    }
  }
  throw new Error('No AI API key available. Add ANTHROPIC_API_KEY or GEMINI_API_KEY to your .env file.');
}

async function runAIModel(content, context) {
  const prompt = `You are an expert fact-checker and misinformation analyst. Analyze the following content using ALL available context.
${context}
CONTENT TO ANALYZE:
"""
${content.slice(0, 3500)}
"""

Return ONLY valid JSON with this structure:
{
  "verdict": "LIKELY FAKE" or "SUSPICIOUS" or "UNVERIFIED" or "CREDIBLE" or "SATIRE",
  "credibility_score": number 0-100,
  "confidence": number 0-100,
  "risk_level": "HIGH" or "MEDIUM" or "LOW" or "SAFE",
  "red_flags": [{"label":"short title","explanation":"one sentence","severity":"high" or "medium" or "low"}],
  "green_flags": [{"label":"short title","explanation":"one sentence"}],
  "metrics": {"sensationalism":0-100,"emotional_manipulation":0-100,"factual_consistency":0-100,"source_credibility":0-100,"writing_quality":0-100},
  "summary": "2-3 sentence explanation",
  "recommendation": "actionable paragraph"
}
Rules: red_flags 0-5, green_flags 0-4, be specific. Return ONLY JSON.`;

  const { text, engine } = await callAI(prompt);
  const parsed = JSON.parse(text);
  parsed._ai_engine = engine;
  return parsed;
}

async function runAIModelLink(url, content, sourceInfo, context) {
  const domain = new URL(url).hostname;

  const prompt = `You are an expert fact-checker. Analyze this URL with ALL provided context.

URL: ${url}
DOMAIN: ${domain}
SOURCE TRUST: ${sourceInfo.score}/100 (${sourceInfo.category}, Bias: ${sourceInfo.bias}) — ${sourceInfo.notes}
${content ? `\nEXTRACTED ARTICLE CONTENT:\n"""\n${content.slice(0, 3000)}\n"""` : '\n(Could not extract article content — analyze based on URL, domain, and context)'}
${context}

Return ONLY valid JSON:
{
  "verdict": "LIKELY TRUE" or "POSSIBLY MISLEADING" or "LIKELY FALSE" or "UNVERIFIED",
  "confidence_score": number 0-100,
  "source_credibility_score": number 0-100,
  "risk_level": "HIGH" or "MEDIUM" or "LOW",
  "main_claim": "one sentence summary",
  "key_claims": [{"claim":"text","status":"verified" or "disputed" or "unverified" or "false","explanation":"reason"}],
  "fact_check_summary": "2-4 sentences",
  "recommendation": "SHARE" or "VERIFY" or "DO NOT SHARE",
  "recommendation_text": "paragraph",
  "source_info": {"domain":"${domain}","reputation":"note","bias":"bias"},
  "misinformation_signals": ["signal1","signal2"]
}
Rules: key_claims 2-5. Use the source trust data provided. Return ONLY JSON.`;

  const { text, engine } = await callAI(prompt);
  const parsed = JSON.parse(text);
  parsed._ai_engine = engine;
  return parsed;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE STEP 4: FACT-CHECK APIs (with fallback)
// ═══════════════════════════════════════════════════════════════════════════════
async function queryNewsAPI(keywords) {
  if (!hasKey(NEWSAPI_KEY)) return { source: 'newsapi', status: 'skipped' };
  try {
    const q = encodeURIComponent(keywords.slice(0, 100));
    const res = await fetchWithTimeout(`https://newsapi.org/v2/everything?q=${q}&sortBy=relevancy&pageSize=5&language=en&apiKey=${NEWSAPI_KEY}`);
    const data = await res.json();
    if (data.status !== 'ok') throw new Error(data.message);
    return { source: 'newsapi', status: 'success', total: data.totalResults || 0,
      articles: (data.articles || []).slice(0, 5).map(a => ({ title: a.title, source: a.source?.name, url: a.url })) };
  } catch (err) { return { source: 'newsapi', status: 'failed', error: err.message }; }
}

async function queryGoogleFactCheck(query) {
  if (!hasKey(GOOGLE_FC_KEY)) return { source: 'google_factcheck', status: 'skipped' };
  try {
    const q = encodeURIComponent(query.slice(0, 200));
    const res = await fetchWithTimeout(`https://factchecktools.googleapis.com/v1alpha1/claims:search?query=${q}&pageSize=5&key=${GOOGLE_FC_KEY}`);
    const data = await res.json();
    return { source: 'google_factcheck', status: 'success',
      claims: (data.claims || []).slice(0, 5).map(c => ({
        text: c.text, claimant: c.claimant,
        reviews: (c.claimReview || []).map(r => ({ publisher: r.publisher?.name, rating: r.textualRating, url: r.url }))
      })) };
  } catch (err) { return { source: 'google_factcheck', status: 'failed', error: err.message }; }
}

async function queryGNews(keywords) {
  if (!hasKey(GNEWS_KEY)) return { source: 'gnews', status: 'skipped' };
  try {
    const q = encodeURIComponent(keywords.slice(0, 100));
    const res = await fetchWithTimeout(`https://gnews.io/api/v4/search?q=${q}&max=5&lang=en&token=${GNEWS_KEY}`);
    const data = await res.json();
    return { source: 'gnews', status: 'success', total: data.totalArticles || 0,
      articles: (data.articles || []).slice(0, 5).map(a => ({ title: a.title, source: a.source?.name, url: a.url })) };
  } catch (err) { return { source: 'gnews', status: 'failed', error: err.message }; }
}

function extractKeywords(text) {
  const stops = new Set(['the','a','an','is','are','was','were','be','been','have','has','had','do','does','did',
    'will','would','shall','should','may','might','must','can','could','not','and','but','or','for','of','at',
    'by','from','in','into','on','to','with','that','this','it','its','i','you','he','she','we','they','all']);
  return text.toLowerCase().replace(/[^a-z0-9\s]/g,' ').split(/\s+/).filter(w => w.length > 3 && !stops.has(w)).slice(0,8).join(' ');
}

function buildExternalContext(newsapi, factcheck, gnews) {
  let ctx = '';
  if (newsapi.status === 'success' && newsapi.total > 0) {
    ctx += `\n\nNEWSAPI CROSS-CHECK (${newsapi.total} results):\n`;
    newsapi.articles.forEach(a => { ctx += `· "${a.title}" — ${a.source}\n`; });
  }
  if (factcheck.status === 'success' && factcheck.claims?.length > 0) {
    ctx += `\n\nGOOGLE FACT-CHECK MATCHES:\n`;
    factcheck.claims.forEach(c => { ctx += `· Claim: "${c.text}" → ${c.reviews?.[0]?.rating || 'N/A'} (${c.reviews?.[0]?.publisher || ''})\n`; });
  }
  if (gnews.status === 'success' && gnews.total > 0) {
    ctx += `\n\nGNEWS COVERAGE (${gnews.total} articles):\n`;
    gnews.articles.forEach(a => { ctx += `· "${a.title}" — ${a.source}\n`; });
  }
  return ctx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROUTE: POST /api/analyze — TEXT ANALYSIS PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════
app.post('/api/analyze', async (req, res) => {
  const { text } = req.body;
  if (!text || text.length < 20) return res.status(400).json({ error: 'Text must be at least 20 characters.' });

  const pipeline = { content_extraction: 'n/a', source_trust: 'n/a', local_ml: 'pending', similarity: 'pending', ai_model: 'pending', fact_check_apis: 'pending' };

  try {
    const keywords = extractKeywords(text);

    // Steps in parallel: fact-check APIs + local ML model + similarity check
    const [newsapi, factcheck, gnews, mlResult, simResult] = await Promise.allSettled([
      queryNewsAPI(keywords), queryGoogleFactCheck(text.slice(0, 200)), queryGNews(keywords),
      callLocalML(text),
      callSimilarityCheck(text)
    ]);
    const na  = newsapi.status   === 'fulfilled' ? newsapi.value   : { status: 'failed' };
    const fc  = factcheck.status === 'fulfilled' ? factcheck.value : { status: 'failed' };
    const gn  = gnews.status     === 'fulfilled' ? gnews.value     : { status: 'failed' };
    const ml  = mlResult.status  === 'fulfilled' ? mlResult.value  : null;
    const sim = simResult.status === 'fulfilled' ? simResult.value : null;

    pipeline.fact_check_apis = 'done';
    pipeline.local_ml  = ml  ? 'done' : (mlServerOnline ? 'failed' : 'skipped');
    pipeline.similarity = sim ? 'done' : (mlServerOnline ? 'failed' : 'skipped');

    // Build context string (includes ML result for AI to consider)
    let mlContext = '';
    if (ml && ml.result) {
      mlContext = `\n\nLOCAL ML MODEL (${ml.model_used || 'Logistic Regression'}, ${ml.model_accuracy || 0}% accuracy):\n`;
      mlContext += `  Prediction: ${ml.result} | Trust: ${ml.trust_score}% | Confidence: ${ml.confidence}%\n`;
    }
    if (sim && sim.verdict) {
      mlContext += `\nSIMILARITY ENGINE (TF-IDF + Cosine):\n`;
      mlContext += `  Verdict: ${sim.verdict} | Max similarity: ${sim.max_pct}% | KB size: ${sim.kb_size} articles\n`;
      if (sim.top_matches && sim.top_matches.length > 0) {
        const m = sim.top_matches[0];
        mlContext += `  Closest match: "${m.title}" (${m.source}, ${m.score*100}% similar)\n`;
      }
    }

    const context = buildExternalContext(na, fc, gn) + mlContext;
    const aiResult = await runAIModel(text, context);
    pipeline.ai_model = 'done';

    // Blend AI credibility score with local ML (if available)
    if (ml && typeof ml.trust_score === 'number') {
      aiResult.credibility_score = Math.round(aiResult.credibility_score * 0.65 + ml.trust_score * 0.35);
      aiResult.local_ml = {
        result:      ml.result,
        trust_score: ml.trust_score,
        confidence:  ml.confidence,
        model_used:  ml.model_used
      };
    }

    // Attach similarity results
    if (sim) {
      aiResult.similarity = {
        verdict:     sim.verdict,
        max_score:   sim.max_score,
        max_pct:     sim.max_pct,
        confidence:  sim.confidence,
        explanation: sim.explanation,
        top_matches: (sim.top_matches || []).slice(0, 3),
        kb_size:     sim.kb_size,
        engine:      sim.engine,
      };
    }

    let sources = 1;
    if (na.status === 'success') sources++;
    if (fc.status === 'success') sources++;
    if (gn.status === 'success') sources++;
    if (ml) sources++;
    if (sim) sources++;

    aiResult.sources_checked = sources;
    aiResult.pipeline = pipeline;
    aiResult._api_status = { ai: 'success', newsapi: na.status, google_factcheck: fc.status, gnews: gn.status, local_ml: pipeline.local_ml, similarity: pipeline.similarity };

    res.json(aiResult);
  } catch (err) {
    console.error('Text pipeline error:', err.message);
    pipeline.ai_model = 'failed';
    res.status(500).json({ error: 'Analysis failed: ' + err.message, pipeline });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// ROUTE: POST /api/analyze-link — FULL URL PIPELINE
// Pipeline: URL → Content Extraction → Source Trust → AI Model → Fact-Check → Merge
// ═══════════════════════════════════════════════════════════════════════════════
app.post('/api/analyze-link', async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: 'URL is required.' });

  let parsedUrl;
  try { parsedUrl = new URL(url); } catch { return res.status(400).json({ error: 'Invalid URL format.' }); }

  const pipeline = { content_extraction: 'pending', source_trust: 'pending', ai_model: 'pending', fact_check_apis: 'pending' };

  try {
    const domain = parsedUrl.hostname;

    // ── STEP 1: Content Extraction ──
    const extracted = await extractContent(url);
    pipeline.content_extraction = extracted.status;

    // ── STEP 2: Source Trust Score ──
    const sourceInfo = getSourceTrustScore(domain);
    pipeline.source_trust = 'done';

    // ── STEP 3: Fact-check APIs (parallel) ──
    const keywords = extractKeywords(
      (extracted.title || '') + ' ' + (extracted.text || '').slice(0, 200) + ' ' + parsedUrl.pathname.replace(/[-_/]/g, ' ')
    );
    const [newsapi, factcheck, gnews] = await Promise.allSettled([
      queryNewsAPI(keywords || domain), queryGoogleFactCheck(extracted.title || keywords || domain), queryGNews(keywords || domain)
    ]);
    const na = newsapi.status === 'fulfilled' ? newsapi.value : { status: 'failed' };
    const fc = factcheck.status === 'fulfilled' ? factcheck.value : { status: 'failed' };
    const gn = gnews.status === 'fulfilled' ? gnews.value : { status: 'failed' };
    pipeline.fact_check_apis = 'done';

    // ── STEP 4: AI Model Analysis + ML + Similarity (all parallel) ──
    const context = buildExternalContext(na, fc, gn);
    const [aiResult, mlResult, simResult] = await Promise.allSettled([
      runAIModelLink(url, extracted.text, sourceInfo, context),
      callLocalML(extracted.text),
      callSimilarityCheck(extracted.text, extracted.title || '')
    ]);
    if (aiResult.status === 'rejected') throw aiResult.reason;
    const ai  = aiResult.value;
    const ml  = mlResult.status  === 'fulfilled' ? mlResult.value  : null;
    const sim = simResult.status === 'fulfilled' ? simResult.value : null;
    pipeline.ai_model   = 'done';
    pipeline.local_ml   = ml  ? 'done' : (mlServerOnline ? 'failed' : 'skipped');
    pipeline.similarity = sim ? 'done' : (mlServerOnline ? 'failed' : 'skipped');

    // ── STEP 5: Merge Pipeline Results ──
    let sources = 1;
    if (na.status === 'success') sources++;
    if (fc.status === 'success') sources++;
    if (gn.status === 'success') sources++;
    if (ml) sources++;
    if (sim) sources++;

    // Blend AI credibility score with source trust database
    if (sourceInfo.matched) {
      ai.source_credibility_score = Math.round(
        ai.source_credibility_score * 0.5 + sourceInfo.score * 0.5
      );
    }

    // Blend with local ML if available
    if (ml && typeof ml.trust_score === 'number') {
      ai.confidence_score = Math.round((ai.confidence_score || 50) * 0.65 + ml.trust_score * 0.35);
      ai.local_ml = {
        result:      ml.result,
        trust_score: ml.trust_score,
        confidence:  ml.confidence,
        model_used:  ml.model_used
      };
    }

    // Attach similarity results
    if (sim) {
      ai.similarity = {
        verdict:     sim.verdict,
        max_score:   sim.max_score,
        max_pct:     sim.max_pct,
        confidence:  sim.confidence,
        explanation: sim.explanation,
        top_matches: (sim.top_matches || []).slice(0, 3),
        kb_size:     sim.kb_size,
        engine:      sim.engine,
      };
    }

    ai.sources_checked = sources;
    ai.pipeline = pipeline;
    ai.source_trust_db = {
      score: sourceInfo.score, category: sourceInfo.category,
      bias: sourceInfo.bias,   notes: sourceInfo.notes, matched: sourceInfo.matched
    };
    ai.content_extracted = {
      status: extracted.status, title: extracted.title || null,
      wordCount: extracted.wordCount, author: extracted.meta?.author || null,
      publishDate: extracted.meta?.publishDate || null
    };
    ai._api_status = { ai: 'success', newsapi: na.status, google_factcheck: fc.status,
      gnews: gn.status, content_extraction: extracted.status, local_ml: pipeline.local_ml, similarity: pipeline.similarity };

    res.json(ai);
  } catch (err) {
    console.error('Link pipeline error:', err.message);
    pipeline.ai_model = 'failed';
    res.status(500).json({ error: 'Analysis failed: ' + err.message, pipeline });
  }
});


// ─── Health ──────────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    apis: {
      anthropic: hasKey(ANTHROPIC_KEY), gemini: hasKey(GEMINI_KEY),
      newsapi: hasKey(NEWSAPI_KEY), google_factcheck: hasKey(GOOGLE_FC_KEY), gnews: hasKey(GNEWS_KEY)
    },
    local_ml: { online: mlServerOnline, server: ML_SERVER }
  });
});

// ─── Standalone Similarity Check (proxied from Node → Python ML server) ──────
// POST /api/similarity  { text, title? }
app.post('/api/similarity', async (req, res) => {
  const { text = '', title = '' } = req.body;
  if (!text || text.trim().length < 20)
    return res.status(400).json({ error: 'text must be at least 20 characters.' });
  const result = await callSimilarityCheck(text.trim(), title.trim());
  if (!result)
    return res.status(503).json({ error: 'Similarity engine offline. Start Python ML server first.' });
  res.json(result);
});

// GET /api/similarity/stats
app.get('/api/similarity/stats', async (req, res) => {
  try {
    const r = await fetchWithTimeout(`${ML_SERVER}/api/similarity/stats`, {}, 4000);
    res.json(await r.json());
  } catch {
    res.status(503).json({ error: 'ML server offline' });
  }
});


// ─── ML Server status proxy ───────────────────────────────────────────────────
app.get('/api/ml/status', async (req, res) => {
  const online = await checkMLServer();
  try {
    if (online) {
      const r = await fetchWithTimeout(`${ML_SERVER}/api/ml/health`, {}, 3000);
      const d = await r.json();
      return res.json({ online: true, ...d });
    }
  } catch {}
  res.json({ online: false, message: 'ML server not running. Start with: python ml_engine/ml_server.py' });
});

// ─── Source Reliability Checker ───────────────────────────────────────────────
// POST /api/check-source  { url, title?, content?, author?, publishDate? }
// GET  /api/check-source?url=...
app.all('/api/check-source', async (req, res) => {
  const body   = req.method === 'GET' ? req.query : req.body;
  const url    = (body.url    || '').trim();
  const title  = (body.title  || '').trim();
  const content= (body.content|| '').slice(0, 3000);
  const author = (body.author || '').trim();
  const publishDate = (body.publishDate || '').trim();

  if (!url) return res.status(400).json({ error: 'url is required' });

  try {
    const result = computeSourceReliability({ url, title, content, author, publishDate });
    res.json({
      ok: true,
      url,
      domain:         result.domain,
      finalScore:     result.finalScore,
      tier:           result.tier,
      tierInfo:       result.tierInfo,
      verdict:        result.verdict,
      recommendation: result.recommendation,
      breakdown:      result.breakdown,
      sensationalHits: result.sensationalHits,
      domainInfo:     result.domainInfo,
      isHttps:        result.isHttps,
      hasAuthor:      result.hasAuthor,
      hasDate:        result.hasDate,
      totalDomains:   Object.keys(SOURCE_TRUST_DB).length,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── Source DB list (for frontend display) ───────────────────────────────────
app.get('/api/source-db', (req, res) => {
  const tier = parseInt(req.query.tier) || null;
  const entries = Object.entries(SOURCE_TRUST_DB)
    .filter(([, v]) => !tier || v.tier === tier)
    .sort((a, b) => b[1].score - a[1].score)
    .map(([domain, v]) => ({ domain, ...v }));
  res.json({ total: entries.length, entries });
});


// ─── Start ───────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log('\n═══════════════════════════════════');
  console.log('  NewsGuard Multi-Stage Pipeline');
  console.log('═══════════════════════════════════');
  console.log(`  Anthropic AI : ${hasKey(ANTHROPIC_KEY) ? '✅' : '⚪ Not set'}`);
  console.log(`  Gemini AI    : ${hasKey(GEMINI_KEY) ? '✅ (fallback)' : '⚪ Not set'}`);
  console.log(`  AI Engine    : ${hasKey(ANTHROPIC_KEY) ? 'Anthropic → Gemini fallback' : hasKey(GEMINI_KEY) ? 'Gemini only' : '❌ No AI key!'}`);
  console.log(`  NewsAPI      : ${hasKey(NEWSAPI_KEY) ? '✅' : '⚪ Skipped'}`);
  console.log(`  Google FC    : ${hasKey(GOOGLE_FC_KEY) ? '✅' : '⚪ Skipped'}`);
  console.log(`  GNews        : ${hasKey(GNEWS_KEY) ? '✅' : '⚪ Skipped'}`);
  console.log(`  Local ML     : Checking port 8000...`);
  console.log(`\n🛡️  http://127.0.0.1:${PORT}/newsguard.html\n`);
  // Async check for local ML server (non-blocking)
  checkMLServer().then(online => {
    if (!online) console.log('  🤖 Local ML Server: ⚪ Not running (optional — start with: python ml_engine/ml_server.py)');
  });
});
