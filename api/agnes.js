// ==========================================================================
// api/agnes.js — Vercel Serverless Function for Agnes AI
// Powered by OpenAI + Tavily Web Search (optional)
// Deployed on Vercel. Reads keys from environment variables.
// ==========================================================================

// --------------------------------------------------------------------------
// 0 · CONFIGURATION
// --------------------------------------------------------------------------
const CONFIG = {
  // LLM Provider — set OPENAI_API_KEY in Vercel env vars
  openaiModel: 'gpt-4o-mini', // or 'gpt-4o', 'gpt-4-turbo'

  // Web Search Provider — set WEB_SEARCH_PROVIDER to 'tavily', 'brave', or 'none'
  webSearchProvider: (process.env.WEB_SEARCH_PROVIDER || 'tavily').toLowerCase(),

  // Search behaviour
  maxSearchResults: 5,
  maxContextChars: 8000, // budget for page context in system prompt
  maxSearchResultChars: 2000, // budget for each search result

  // LLM behaviour
  temperature: 0.4,
  maxTokens: 700,
};

// --------------------------------------------------------------------------
// 1 · CORE HANDLER
// --------------------------------------------------------------------------
export default async function handler(req, res) {
  // ----------------------------------------------------------------------
  // 1.1 · CORS & Method Guard
  // ----------------------------------------------------------------------
  res.setHeader('Access-Control-Allow-Origin', '*'); // Change to your domain in production
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  // ----------------------------------------------------------------------
  // 1.2 · Parse Request
  // ----------------------------------------------------------------------
  let { message, context } = req.body || {};

  if (!message || typeof message !== 'string' || message.trim().length === 0) {
    return res.status(400).json({ reply: 'No message received. Please type a question.' });
  }

  message = message.trim();

  // ----------------------------------------------------------------------
  // 1.3 · Validate API Key
  // ----------------------------------------------------------------------
  const openaiKey = process.env.OPENAI_API_KEY;
  if (!openaiKey) {
    console.error('[Agnes] Missing OPENAI_API_KEY environment variable.');
    return res.status(500).json({
      reply: 'Server not configured. Please WhatsApp +91 62610 31710 for instant help.',
    });
  }

  // ----------------------------------------------------------------------
  // 1.4 · Optional: Perform Web Search
  // ----------------------------------------------------------------------
  let searchResults = '';
  let searchUsed = false;

  if (CONFIG.webSearchProvider !== 'none') {
    try {
      const searchResponse = await performWebSearch(message);
      if (searchResponse && searchResponse.results.length > 0) {
        searchResults = formatSearchResults(searchResponse);
        searchUsed = true;
        console.log(`[Agnes] Web search returned ${searchResponse.results.length} results.`);
      }
    } catch (err) {
      console.warn('[Agnes] Web search failed, continuing without it:', err.message);
    }
  }

  // ----------------------------------------------------------------------
  // 1.5 · Build System Prompt
  // ----------------------------------------------------------------------
  const systemPrompt = buildSystemPrompt(context, searchResults);

  // ----------------------------------------------------------------------
  // 1.6 · Call LLM
  // ----------------------------------------------------------------------
  try {
    const llmResponse = await callLLM(message, systemPrompt, openaiKey);
    return res.status(200).json({ reply: llmResponse, searchUsed });
  } catch (err) {
    console.error('[Agnes] LLM call failed:', err);

    // Fallback: if we have search results, return a formatted answer
    if (searchResults) {
      return res.status(200).json({
        reply:
          'Here\'s what I found from a web search:\n\n' +
          searchResults +
          '\n\nFor more details, WhatsApp +91 62610 31710.',
        searchUsed: true,
        fallback: true,
      });
    }

    return res.status(500).json({
      reply: 'Agnes is offline right now. Please WhatsApp +91 62610 31710 for instant help.',
      searchUsed: false,
    });
  }
}

// --------------------------------------------------------------------------
// 2 · WEB SEARCH LAYER
// --------------------------------------------------------------------------

/**
 * Performs a web search using the configured provider.
 * Returns { query, results: [{ title, url, content, score }] }
 */
async function performWebSearch(query) {
  const provider = CONFIG.webSearchProvider;
  const apiKey = process.env.WEB_SEARCH_API_KEY;

  if (!apiKey) {
    console.warn('[Agnes] Missing WEB_SEARCH_API_KEY. Skipping web search.');
    return { query, results: [] };
  }

  switch (provider) {
    case 'tavily':
      return await searchWithTavily(query, apiKey);
    case 'brave':
      return await searchWithBrave(query, apiKey);
    default:
      console.warn(`[Agnes] Unknown search provider: ${provider}`);
      return { query, results: [] };
  }
}

/**
 * Tavily Search API
 * Docs: https://docs.tavily.com/docs/rest-api/api-reference
 */
async function searchWithTavily(query, apiKey) {
  const response = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      api_key: apiKey,
      query: query,
      search_depth: 'basic', // 'basic' or 'advanced'
      include_answer: false, // we'll let the LLM synthesize
      include_raw_content: false,
      max_results: CONFIG.maxSearchResults,
      include_domains: [],
      exclude_domains: [],
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Tavily error ${response.status}: ${errorText}`);
  }

  const data = await response.json();

  // Normalize results
  const results = (data.results || []).map((r) => ({
    title: r.title || 'Untitled',
    url: r.url || '',
    content: r.content || r.snippet || '',
    score: r.score || 0,
  }));

  return { query, results, answer: data.answer || '' };
}

/**
 * Brave Search API
 * Docs: https://brave.com/search/api/
 */
async function searchWithBrave(query, apiKey) {
  const url = new URL('https://api.search.brave.com/res/v1/web/search');
  url.searchParams.set('q', query);
  url.searchParams.set('count', CONFIG.maxSearchResults);
  url.searchParams.set('safesearch', 'moderate');
  url.searchParams.set('text_decorations', 'false');

  const response = await fetch(url.toString(), {
    method: 'GET',
    headers: {
      Accept: 'application/json',
      'Accept-Encoding': 'gzip',
      'X-Subscription-Token': apiKey,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Brave error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const webResults = data.web?.results || [];

  const results = webResults.map((r) => ({
    title: r.title || 'Untitled',
    url: r.url || '',
    content: r.description || r.snippet || '',
    score: 0.5, // Brave doesn't expose a score
  }));

  return { query, results };
}

/**
 * Formats search results into a clean string for the LLM prompt.
 */
function formatSearchResults({ query, results }) {
  if (!results.length) return '';

  let formatted = `Web search results for "${query}":\n\n`;

  results.forEach((r, i) => {
    const snippet = r.content.slice(0, CONFIG.maxSearchResultChars);
    formatted += `${i + 1}. **${r.title}**\n`;
    formatted += `   URL: ${r.url}\n`;
    formatted += `   ${snippet}${r.content.length > CONFIG.maxSearchResultChars ? '…' : ''}\n\n`;
  });

  return formatted.trim();
}

// --------------------------------------------------------------------------
// 3 · PROMPT BUILDER
// --------------------------------------------------------------------------

function buildSystemPrompt(pageContext, searchResults) {
  let prompt = `You are **Agnes**, the AI strategist for **Kalki-Intelligence** — a full-stack digital growth agency.

Your job is to help visitors understand our services, pricing, case studies, ROI, and process. You are:
- **Concise** — answer in 2–4 sentences unless asked for detail.
- **Confident** — you know our offerings inside out.
- **Helpful** — guide users toward the right next step.
- **Honest** — never invent prices, metrics, or guarantees. Use only what's in the CONTEXT below.

## Services We Offer (Summary)
- **SEO** — from ₹15,000/mo
- **AEO** (Answer Engine Optimization) — from ₹20,000/mo
- **GEO** (Generative Engine Optimization) — from ₹25,000/mo
- **Web Development** — from ₹35,000 one-time
- **AI Bots** (WhatsApp + Web) — from ₹12,000/mo
- **Paid Ads** (Google + Meta + YouTube) — from ₹15,000/mo + ad spend

## Bundles
- **Starter Bundle** — ₹35,000/mo
- **Growth Bundle** — ₹75,000/mo (most popular)
- **Domination Bundle** — ₹1,80,000/mo

## Contact
- WhatsApp: +91 62610 31710
- Email: ceo@kalki-intelligence.in
- Website: kalki-intelligence.in

## Tone & Style
- Use markdown: **bold**, *italic*, bullet lists.
- Keep answers scannable.
- If the user asks about something outside our scope, gently steer them back to our services.
- If you're unsure, say "I'm not certain, but you can WhatsApp +91 62610 31710 for a definitive answer."

`;

  // Inject page context (from the client-side indexer)
  if (pageContext && pageContext.trim().length > 0) {
    prompt += `\n## PAGE CONTEXT (from the live proposal page)\n`;
    prompt += pageContext.slice(0, CONFIG.maxContextChars);
    prompt += `\n`;
  }

  // Inject web search results (if any)
  if (searchResults && searchResults.trim().length > 0) {
    prompt += `\n## WEB SEARCH RESULTS (real-time, use only if relevant to the user's question)\n`;
    prompt += searchResults;
    prompt += `\n`;
  }

  prompt += `\n## FINAL INSTRUCTION\nAnswer the user's question using the context above. If the context doesn't contain the answer, say so politely and suggest contacting us on WhatsApp (+91 62610 31710).`;

  return prompt;
}

// --------------------------------------------------------------------------
// 4 · LLM CALL
// --------------------------------------------------------------------------

async function callLLM(userMessage, systemPrompt, apiKey) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: CONFIG.openaiModel,
      temperature: CONFIG.temperature,
      max_tokens: CONFIG.maxTokens,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage },
      ],
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI error ${response.status}: ${errorText}`);
  }

  const data = await response.json();

  const reply = data.choices?.[0]?.message?.content;
  if (!reply) {
    throw new Error('OpenAI returned an empty response.');
  }

  return reply.trim();
}

// --------------------------------------------------------------------------
// 5 · EXPORT FOR TESTING (optional)
// --------------------------------------------------------------------------
// You can import these in a test file if needed.
export const _test = {
  searchWithTavily,
  searchWithBrave,
  formatSearchResults,
  buildSystemPrompt,
  callLLM,
};
