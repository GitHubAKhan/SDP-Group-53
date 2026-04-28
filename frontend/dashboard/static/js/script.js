let sectorChart = null;
let performanceChart = null;
let currentTimePeriod = '3M';
let currentSortBy = 'allocation';
let allPositions = [];
let allSectors = {};
let performanceData = [];
const FINNHUB_API_KEY = 'd6etk0hr01qvn4o1162gd6etk0hr01qvn4o11630'
const MARKETAUX_API_KEY = 'P0z2oBlZVX9t5Uq131ZVUQ3L69x8Tt5Qr8dn8pol'; 
const API_BASE_URL = 'https://hdo6lukv03.execute-api.us-east-1.amazonaws.com/prod';

// Helper function to call Lambda via API Gateway
async function callLambdaAPI(endpoint, apiKey, apiSecret) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey, apiSecret })
    });
    
    if (!response.ok) {
        throw new Error(`API error (${endpoint}): ${response.status}`);
    }
    
    const data = await response.json();
    
    // Parse the body string (Lambda double-encodes)
    return JSON.parse(data.body);
}

// ===== THEME TOGGLE =====
function toggleTheme() {
    const html = document.documentElement;
    const icon = document.getElementById('theme-icon');
    const currentTheme = html.getAttribute('data-theme');

    if (currentTheme === 'dark') {
        html.setAttribute('data-theme', 'light');
        icon.className = 'fas fa-moon';
        localStorage.setItem('theme', 'light');
    } else {
        html.setAttribute('data-theme', 'dark');
        icon.className = 'fas fa-sun';
        localStorage.setItem('theme', 'dark');
    }

    // Recreate charts with new theme
    if (sectorChart) {
        createSectorPieChart();
    }
    if (performanceChart) {
        createPerformanceLineChart();
    }
}

// Load saved theme
window.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    if (savedTheme === 'dark') {
        document.getElementById('theme-icon').className = 'fas fa-sun';
    }
});

// ===== LOGIN =====
function handleLogin(event) {
    event.preventDefault();
    const apiKey = document.getElementById('api-key').value;
    const apiSecret = document.getElementById('api-secret').value;

    sessionStorage.setItem('alpaca_key', apiKey);
    sessionStorage.setItem('alpaca_secret', apiSecret);

    document.getElementById('login-page').style.display = 'none';
    document.getElementById('main-dashboard').style.display = 'block';

    loadDashboardData();
}

function logout() {
    if (confirm('Are you sure you want to logout?')) {
        sessionStorage.clear();
        location.reload();
    }
}

// ===== TAB SWITCHING =====
function switchTab(tabName) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.nav-item').classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName + '-tab').classList.add('active');

    // Update title
    const titles = {
        'dashboard': 'Dashboard',
        'positions': 'Positions',
        'sectors': 'Sectors',
        'news': 'News Feed'
    };
    document.getElementById('page-title').textContent = titles[tabName];

    // Load news feed when news tab is opened
    if (tabName === 'news') {
        loadNewsFeed();
    }
}

//Revised loadDashboardData to handle API Gateway and Lambda response formats, with enhanced error handling and logging
async function loadDashboardData() {
  try {
    const apiKey = sessionStorage.getItem('alpaca_key');
    const apiSecret = sessionStorage.getItem('alpaca_secret');

    console.log("Fetching account data...");
    console.log("API Key from session:", apiKey);
    console.log("API Secret from session:", apiSecret); 
    
    // --- Fetch account data ---
    const accountResponse = await fetch(`${API_BASE_URL}/account`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey, apiSecret })
    });
    
    if (!accountResponse.ok) {
      throw new Error(`Account API error: ${accountResponse.status}`);
    }
    
    const accountData = await accountResponse.json();
    console.log("Raw account data:", accountData);
    
    // Parse Lambda response - handle both formats
    let account;
    if (accountData.body) {
      // Lambda proxy integration - body is a JSON string
      if (typeof accountData.body === 'string') {
        account = JSON.parse(accountData.body);
      } else {
        account = accountData.body;
      }
    } else {
      // Direct response
      account = accountData;
    }
    
    console.log("Parsed account:", account);

    // --- Fetch positions data ---
    console.log("Fetching positions data...");
    
    const positionsResponse = await fetch(`${API_BASE_URL}/positions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey, apiSecret })
    });
    
    if (!positionsResponse.ok) {
      throw new Error(`Positions API error: ${positionsResponse.status}`);
    }
    
    const positionsData = await positionsResponse.json();
    console.log("Raw positions data:", positionsData);
    
    // Parse Lambda response - handle both formats
    let positions;
    if (positionsData.body) {
      // Lambda proxy integration - body is a JSON string
      if (typeof positionsData.body === 'string') {
        positions = JSON.parse(positionsData.body);
      } else {
        positions = positionsData.body;
      }
    } else {
      // Direct response
      positions = positionsData;
    }
    
    console.log("Parsed positions:", positions);
    console.log("Positions is array?", Array.isArray(positions));
    
    // Safety check - make sure positions is an array
    if (!Array.isArray(positions)) {
      console.error("ERROR: Positions is not an array!", positions);
      alert("Error: Received invalid positions data. Check console for details.");
      return;
    }

    // Fetch 1W and 1M history in parallel for accurate Performance Overview
    const apiKey2 = sessionStorage.getItem('alpaca_key');
    const apiSecret2 = sessionStorage.getItem('alpaca_secret');
    const [weekHistory, monthHistory] = await Promise.all([
        fetch(`https://paper-api.alpaca.markets/v2/account/portfolio/history?timeframe=1D&period=1W`, { headers: { 'APCA-API-KEY-ID': apiKey2, 'APCA-API-SECRET-KEY': apiSecret2 } }).then(r => r.json()).catch(() => null),
        fetch(`https://paper-api.alpaca.markets/v2/account/portfolio/history?timeframe=1D&period=1M`, { headers: { 'APCA-API-KEY-ID': apiKey2, 'APCA-API-SECRET-KEY': apiSecret2 } }).then(r => r.json()).catch(() => null),
    ]);

    // Render results
    populateDashboard(account, positions, weekHistory, monthHistory);
    populatePositionsTable(positions);
    allPositions = positions;
    allSectors = await groupBySector(positions);
    populateSectorsList(allSectors);
    createSectorPieChart();
    createPerformanceLineChart();

  } catch (error) {
    console.error("Error loading dashboard data:", error);
    alert(`Failed to load dashboard: ${error.message}\n\nCheck browser console for details.`);
  }
}

// ===== DASHBOARD =====
function populateDashboard(account, positions, weekHistory, monthHistory) {
    const portfolioValue = parseFloat(account.portfolio_value);
    const initialCapital = 100000;
    const totalReturn = ((portfolioValue - initialCapital) / initialCapital) * 100;
    const dollarGain = portfolioValue - initialCapital;

    // Daily change: use Alpaca's last_equity (portfolio value at previous close) as the baseline.
    // Summing unrealized_pl across positions is wrong — it reflects gain since each position was
    // opened, not since today's market open.
    const lastEquity = parseFloat(account.last_equity || account.last_portfolio_value || portfolioValue);
    const totalUnrealizedPL = portfolioValue - lastEquity;
    const todayChangePC = lastEquity > 0 ? (totalUnrealizedPL / lastEquity) * 100 : 0;

    // Find best and worst performers
    const sorted = [...positions].sort((a, b) =>
        parseFloat(b.unrealized_plpc || 0) - parseFloat(a.unrealized_plpc || 0)
    );
    const bestPerformer = sorted[0];
    const worstPerformer = sorted[sorted.length - 1];

    const metricsHTML = `
        <div class="metric-card">
            <p class="metric-label">Total Value of Fund</p>
            <p class="metric-value">$${portfolioValue.toLocaleString('en-US', {minimumFractionDigits: 2})}</p>
            <p class="metric-change ${dollarGain >= 0 ? 'positive' : 'negative'}">
                <i class="fas fa-arrow-${dollarGain >= 0 ? 'up' : 'down'}"></i>
                ${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%
            </p>
        </div>
        <div class="metric-card">
            <p class="metric-label">Total Profit</p>
            <p class="metric-value">${dollarGain >= 0 ? '+' : ''}$${Math.abs(dollarGain).toLocaleString('en-US', {minimumFractionDigits: 2})}</p>
            <p class="metric-change ${dollarGain >= 0 ? 'positive' : 'negative'}">
                <i class="fas fa-arrow-${dollarGain >= 0 ? 'up' : 'down'}"></i>
                ${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%
            </p>
        </div>
        <div class="metric-card">
            <p class="metric-label">Daily Change</p>
            <p class="metric-value">${totalUnrealizedPL >= 0 ? '+' : ''}$${totalUnrealizedPL.toLocaleString('en-US', {minimumFractionDigits: 2})}</p>
            <p class="metric-change ${totalUnrealizedPL >= 0 ? 'positive' : 'negative'}">
                <i class="fas fa-arrow-${totalUnrealizedPL >= 0 ? 'up' : 'down'}"></i>
                ${todayChangePC.toFixed(2)}%
            </p>
        </div>
        <div class="metric-card">
            <p class="metric-label">Best Performer</p>
            <p class="metric-value">${bestPerformer.symbol}</p>
            <p class="metric-change positive">
                <i class="fas fa-arrow-up"></i>
                +${(parseFloat(bestPerformer.unrealized_plpc) * 100).toFixed(2)}%
            </p>
        </div>
        <div class="metric-card">
            <p class="metric-label">Worst Performer</p>
            <p class="metric-value">${worstPerformer.symbol}</p>
            <p class="metric-change negative">
                <i class="fas fa-arrow-down"></i>
                ${(parseFloat(worstPerformer.unrealized_plpc) * 100).toFixed(2)}%
            </p>
        </div>
    `;
    document.getElementById('dashboard-metrics').innerHTML = metricsHTML;

    // Performance stats — calculated from real Alpaca history.
    // Compare FIRST non-zero equity (period open) vs LAST non-zero equity in the same
    // history array, so stat cards and the graph always draw from the same data.
    function calcChange(history) {
        if (!history || !history.equity || history.equity.length < 2) return null;
        const startValue = history.equity.find(v => v > 0);
        if (!startValue) return null;
        // Walk backward to find the most recent non-zero close
        const endValue = [...history.equity].reverse().find(v => v > 0);
        if (!endValue) return null;
        const dollar = Math.round((endValue - startValue) * 100) / 100;
        const pct = (dollar / startValue) * 100;
        return { dollar, pct };
    }

    const weekChange = calcChange(weekHistory);
    const monthChange = calcChange(monthHistory);

    function renderChange(change, label) {
        if (!change) return `
            <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">${label}</p>
            <p style="font-size: 28px; font-weight: 700; margin-bottom: 4px; color: var(--text-secondary);">N/A</p>
            <p style="font-size: 14px; color: var(--text-secondary);">No data available</p>`;
        const color = change.dollar >= 0 ? 'var(--positive)' : 'var(--negative)';
        const arrow = change.dollar >= 0 ? 'up' : 'down';
        const sign = change.dollar >= 0 ? '+' : '';
        return `
            <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">${label}</p>
            <p style="font-size: 28px; font-weight: 700; margin-bottom: 4px; color: ${color};">${sign}$${Math.abs(change.dollar).toLocaleString('en-US', {minimumFractionDigits: 2})}</p>
            <p style="font-size: 14px; color: ${color};"><i class="fas fa-arrow-${arrow}"></i> ${sign}${change.pct.toFixed(2)}%</p>`;
    }

    const statsHTML = `
        <div style="padding: 20px;">
            <div style="margin-bottom: 24px;">${renderChange(weekChange, 'Weekly Change')}</div>
            <div style="margin-bottom: 24px;">${renderChange(monthChange, 'Monthly Change')}</div>
            <div>
                <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">All Time</p>
                <p style="font-size: 28px; font-weight: 700; margin-bottom: 4px; color: ${dollarGain >= 0 ? 'var(--positive)' : 'var(--negative)'};">${dollarGain >= 0 ? '+' : ''}$${Math.abs(dollarGain).toLocaleString('en-US', {minimumFractionDigits: 2})}</p>
                <p style="font-size: 14px; color: ${dollarGain >= 0 ? 'var(--positive)' : 'var(--negative)'};">
                    <i class="fas fa-arrow-${dollarGain >= 0 ? 'up' : 'down'}"></i>
                    ${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%
                </p>
            </div>
        </div>
    `;
    document.getElementById('performance-stats').innerHTML = statsHTML;
}

// ===== POSITIONS TABLE =====
function populatePositionsTable(positions) {
    const sorted = positions.sort((a, b) => parseFloat(b.market_value) - parseFloat(a.market_value));

    let tableHTML = `
        <thead>
            <tr>
                <th>Company</th>
                <th class="right">Shares</th>
                <th class="right">Price</th>
                <th class="right">Market Value</th>
                <th class="right">Gain/Loss</th>
                <th class="right">Return %</th>
            </tr>
        </thead>
        <tbody>
    `;

    sorted.forEach(pos => {
        const unrealizedPL = parseFloat(pos.unrealized_pl || 0);
        const unrealizedPLPC = parseFloat(pos.unrealized_plpc || 0) * 100;
        const gainClass = unrealizedPL >= 0 ? 'positive' : 'negative';
        const logoColor = `hsl(${pos.symbol.charCodeAt(0) * 137.5}, 70%, 55%)`;

        tableHTML += `
            <tr>
                <td>
                    <div class="stock-cell">
                        <div class="stock-logo" style="background: ${logoColor};">
                            <img src="https://assets.parqet.com/logos/symbol/${pos.symbol}?format=jpg"
                                 onerror="handleLogoError(this, '${pos.symbol}')">
                        </div>
                        <div class="stock-info">
                            <span class="stock-symbol">${pos.symbol}</span>
                            <span class="stock-name">${getCompanyName(pos.symbol)}</span>
                        </div>
                    </div>
                </td>
                <td class="right">${parseFloat(pos.qty).toLocaleString()}</td>
                <td class="right">$${parseFloat(pos.current_price).toFixed(2)}</td>
                <td class="right">$${parseFloat(pos.market_value).toLocaleString('en-US', {minimumFractionDigits: 2})}</td>
                <td class="right ${gainClass}">${unrealizedPL >= 0 ? '+' : ''}$${unrealizedPL.toFixed(2)}</td>
                <td class="right ${gainClass}">${unrealizedPL >= 0 ? '+' : ''}${unrealizedPLPC.toFixed(2)}%</td>
            </tr>
        `;
    });

    tableHTML += `</tbody>`;
    document.getElementById('positions-table').innerHTML = tableHTML;
}

function getCompanyDomain(symbol) {
    const domains = {
        'PLTR': 'palantir.com', 'APP': 'applovin.com', 'AVGO': 'broadcom.com',
        'HOOD': 'robinhood.com', 'IBKR': 'interactivebrokers.com', 'NFLX': 'netflix.com',
        'COIN': 'coinbase.com', 'SCHW': 'schwab.com', 'GS': 'goldmansachs.com'
    };
    return domains[symbol] || `${symbol.toLowerCase()}.com`;
}

// Logo fallback chain: Parqet â†’ Clearbit â†’ text initials
function handleLogoError(imgEl, symbol) {
    const tried = imgEl.dataset.tried || '';
    if (!tried.includes('clearbit')) {
        imgEl.dataset.tried = 'clearbit';
        imgEl.src = `https://logo.clearbit.com/${getCompanyDomain(symbol)}`;
    } else {
        const initials = symbol.substring(0, 2).toUpperCase();
        imgEl.parentElement.innerHTML = `<span style="font-size:11px;font-weight:700;color:#fff;">${initials}</span>`;
    }
}

function getCompanyName(symbol) {
    const names = {
        'PLTR': 'Palantir Technologies', 'APP': 'AppLovin Corp', 'AVGO': 'Broadcom Inc',
        'HOOD': 'Robinhood Markets', 'IBKR': 'Interactive Brokers', 'NFLX': 'Netflix Inc',
        'COIN': 'Coinbase Global', 'SCHW': 'Charles Schwab', 'GS': 'Goldman Sachs',
        'TPR': 'Tapestry Inc', 'VST': 'Vistra Corp', 'RCL': 'Royal Caribbean',
        'UAL': 'United Airlines', 'DASH': 'DoorDash Inc', 'AXON': 'Axon Enterprise',
        'GEV': 'GE Vernova', 'JBL': 'Jabil Inc', 'HWM': 'Howmet Aerospace'
    };
    return names[symbol] || symbol;
}

// ===== SECTORS ======

// Full S&P 500 sector map (GICS classifications)
const SP500_SECTOR_MAP = {
    // â”€â”€ Technology â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'AVGO': 'Technology', 'AMD': 'Technology', 'ORCL': 'Technology',
    'CRM': 'Technology', 'ACN': 'Technology', 'CSCO': 'Technology',
    'IBM': 'Technology', 'ADBE': 'Technology', 'INTU': 'Technology',
    'NOW': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'AMAT': 'Technology', 'LRCX': 'Technology', 'KLAC': 'Technology',
    'MU': 'Technology', 'INTC': 'Technology', 'ADI': 'Technology',
    'MCHP': 'Technology', 'NXPI': 'Technology', 'SWKS': 'Technology',
    'MPWR': 'Technology', 'MRVL': 'Technology', 'ON': 'Technology',
    'TER': 'Technology', 'KEYS': 'Technology', 'ENPH': 'Technology',
    'FTNT': 'Technology', 'PANW': 'Technology', 'CRWD': 'Technology',
    'SNPS': 'Technology', 'CDNS': 'Technology', 'ANSS': 'Technology',
    'PTC': 'Technology', 'EPAM': 'Technology', 'CTSH': 'Technology',
    'IT': 'Technology', 'GLW': 'Technology', 'APH': 'Technology',
    'TEL': 'Technology', 'JBL': 'Technology', 'STX': 'Technology',
    'WDC': 'Technology', 'NTAP': 'Technology', 'HPQ': 'Technology',
    'HPE': 'Technology', 'DELL': 'Technology', 'SMCI': 'Technology',
    'PLTR': 'Technology', 'APP': 'Technology', 'NET': 'Technology',
    'DDOG': 'Technology', 'SNOW': 'Technology', 'ZS': 'Technology',
    'OKTA': 'Technology', 'TEAM': 'Technology', 'HUBS': 'Technology',
    'WDAY': 'Technology', 'VEEV': 'Technology', 'ARM': 'Technology',
    'ANET': 'Technology', 'NTNX': 'Technology', 'PSTG': 'Technology',

    // â”€â”€ Communication Services â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
    'META': 'Communication Services', 'NFLX': 'Communication Services',
    'DIS': 'Communication Services', 'CMCSA': 'Communication Services',
    'TMUS': 'Communication Services', 'VZ': 'Communication Services',
    'T': 'Communication Services', 'CHTR': 'Communication Services',
    'TTWO': 'Communication Services', 'EA': 'Communication Services',
    'RBLX': 'Communication Services', 'MTCH': 'Communication Services',
    'WBD': 'Communication Services', 'PARA': 'Communication Services',
    'FOX': 'Communication Services', 'FOXA': 'Communication Services',
    'NWS': 'Communication Services', 'NWSA': 'Communication Services',
    'OMC': 'Communication Services', 'IPG': 'Communication Services',
    'SNAP': 'Communication Services', 'PINS': 'Communication Services',
    'RDDT': 'Communication Services', 'SPOT': 'Communication Services',
    'TTD': 'Communication Services', 'LYV': 'Communication Services',

    // â”€â”€ Financials â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'BRK.B': 'Financials', 'JPM': 'Financials', 'BAC': 'Financials',
    'WFC': 'Financials', 'GS': 'Financials', 'MS': 'Financials',
    'C': 'Financials', 'AXP': 'Financials', 'BLK': 'Financials',
    'SCHW': 'Financials', 'USB': 'Financials', 'PNC': 'Financials',
    'TFC': 'Financials', 'COF': 'Financials', 'DFS': 'Financials',
    'SYF': 'Financials', 'AIG': 'Financials', 'MET': 'Financials',
    'PRU': 'Financials', 'AFL': 'Financials', 'ALL': 'Financials',
    'PGR': 'Financials', 'TRV': 'Financials', 'CB': 'Financials',
    'HIG': 'Financials', 'MMC': 'Financials', 'AON': 'Financials',
    'WTW': 'Financials', 'BX': 'Financials', 'KKR': 'Financials',
    'APO': 'Financials', 'CG': 'Financials', 'ARES': 'Financials',
    'BK': 'Financials', 'STT': 'Financials', 'NTRS': 'Financials',
    'FDS': 'Financials', 'MSCI': 'Financials', 'SPGI': 'Financials',
    'MCO': 'Financials', 'ICE': 'Financials', 'CME': 'Financials',
    'CBOE': 'Financials', 'NDAQ': 'Financials', 'FIS': 'Financials',
    'FISV': 'Financials', 'GPN': 'Financials', 'PYPL': 'Financials',
    'V': 'Financials', 'MA': 'Financials', 'IBKR': 'Financials',
    'HOOD': 'Financials', 'COIN': 'Financials', 'SOFI': 'Financials',
    'AFRM': 'Financials', 'LC': 'Financials', 'CFG': 'Financials',
    'FITB': 'Financials', 'KEY': 'Financials', 'RF': 'Financials',
    'HBAN': 'Financials', 'MTB': 'Financials', 'ZION': 'Financials',
    'CMA': 'Financials', 'WAL': 'Financials', 'FHN': 'Financials',

    // â”€â”€ Health Care â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'LLY': 'Health Care', 'UNH': 'Health Care', 'JNJ': 'Health Care',
    'ABBV': 'Health Care', 'MRK': 'Health Care', 'TMO': 'Health Care',
    'ABT': 'Health Care', 'DHR': 'Health Care', 'PFE': 'Health Care',
    'AMGN': 'Health Care', 'BMY': 'Health Care', 'GILD': 'Health Care',
    'VRTX': 'Health Care', 'REGN': 'Health Care', 'BIIB': 'Health Care',
    'MRNA': 'Health Care', 'ILMN': 'Health Care', 'IDXX': 'Health Care',
    'IQV': 'Health Care', 'CRL': 'Health Care', 'IQVIA': 'Health Care',
    'SYK': 'Health Care', 'BSX': 'Health Care', 'MDT': 'Health Care',
    'EW': 'Health Care', 'ISRG': 'Health Care', 'ZBH': 'Health Care',
    'BDX': 'Health Care', 'BAX': 'Health Care', 'PODD': 'Health Care',
    'DXCM': 'Health Care', 'HOLX': 'Health Care', 'ALGN': 'Health Care',
    'RMD': 'Health Care', 'GEHC': 'Health Care', 'HSIC': 'Health Care',
    'MCK': 'Health Care', 'CAH': 'Health Care', 'CVS': 'Health Care',
    'CI': 'Health Care', 'HUM': 'Health Care', 'CNC': 'Health Care',
    'MOH': 'Health Care', 'ELV': 'Health Care', 'HCA': 'Health Care',
    'UHS': 'Health Care', 'THC': 'Health Care', 'WAT': 'Health Care',
    'A': 'Health Care', 'MTD': 'Health Care', 'RVTY': 'Health Care',
    'NBIX': 'Health Care', 'EXAS': 'Health Care', 'INCY': 'Health Care',

    // â”€â”€ Consumer Discretionary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
    'TJX': 'Consumer Discretionary', 'BURL': 'Consumer Discretionary',
    'ROST': 'Consumer Discretionary', 'ULTA': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'CMG': 'Consumer Discretionary', 'YUM': 'Consumer Discretionary',
    'DPZ': 'Consumer Discretionary', 'DRI': 'Consumer Discretionary',
    'EAT': 'Consumer Discretionary', 'TXRH': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'LULU': 'Consumer Discretionary',
    'PVH': 'Consumer Discretionary', 'RL': 'Consumer Discretionary',
    'TPR': 'Consumer Discretionary', 'VFC': 'Consumer Discretionary',
    'HBI': 'Consumer Discretionary', 'DECK': 'Consumer Discretionary',
    'SKX': 'Consumer Discretionary', 'UAA': 'Consumer Discretionary',
    'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary',
    'TSCO': 'Consumer Discretionary', 'AZO': 'Consumer Discretionary',
    'ORLY': 'Consumer Discretionary', 'AAP': 'Consumer Discretionary',
    'BBY': 'Consumer Discretionary', 'W': 'Consumer Discretionary',
    'RCL': 'Consumer Discretionary', 'CCL': 'Consumer Discretionary',
    'NCLH': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary',
    'HLT': 'Consumer Discretionary', 'H': 'Consumer Discretionary',
    'MGM': 'Consumer Discretionary', 'WYNN': 'Consumer Discretionary',
    'LVS': 'Consumer Discretionary', 'CZR': 'Consumer Discretionary',
    'DASH': 'Consumer Discretionary', 'UBER': 'Consumer Discretionary',
    'LYFT': 'Consumer Discretionary', 'ABNB': 'Consumer Discretionary',
    'BKNG': 'Consumer Discretionary', 'EXPE': 'Consumer Discretionary',
    'TRIP': 'Consumer Discretionary', 'UAL': 'Consumer Discretionary',
    'DAL': 'Consumer Discretionary', 'LUV': 'Consumer Discretionary',
    'AAL': 'Consumer Discretionary', 'ALK': 'Consumer Discretionary',
    'APTV': 'Consumer Discretionary', 'NVR': 'Consumer Discretionary',
    'PHM': 'Consumer Discretionary', 'DHI': 'Consumer Discretionary',
    'LEN': 'Consumer Discretionary', 'TOL': 'Consumer Discretionary',
    'EL': 'Consumer Discretionary', 'RIVN': 'Consumer Discretionary',
    'LCID': 'Consumer Discretionary',

    // â”€â”€ Consumer Staples â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples',
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'BTI': 'Consumer Staples',
    'MDLZ': 'Consumer Staples', 'GIS': 'Consumer Staples',
    'K': 'Consumer Staples', 'CPB': 'Consumer Staples',
    'HRL': 'Consumer Staples', 'CAG': 'Consumer Staples',
    'SJM': 'Consumer Staples', 'MKC': 'Consumer Staples',
    'CL': 'Consumer Staples', 'CHD': 'Consumer Staples',
    'CLX': 'Consumer Staples', 'KMB': 'Consumer Staples',
    'EL': 'Consumer Staples', 'KVUE': 'Consumer Staples',
    'DLTR': 'Consumer Staples', 'DG': 'Consumer Staples',
    'KR': 'Consumer Staples', 'SYY': 'Consumer Staples',
    'BJ': 'Consumer Staples', 'MNST': 'Consumer Staples',
    'STZ': 'Consumer Staples', 'TAP': 'Consumer Staples',
    'BUD': 'Consumer Staples',

    // â”€â”€ Industrials â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'GE': 'Industrials', 'GEV': 'Industrials', 'HON': 'Industrials',
    'MMM': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials',
    'NOC': 'Industrials', 'GD': 'Industrials', 'BA': 'Industrials',
    'TDG': 'Industrials', 'HEI': 'Industrials', 'TXT': 'Industrials',
    'HWM': 'Industrials', 'SPR': 'Industrials', 'AXON': 'Industrials',
    'CAT': 'Industrials', 'DE': 'Industrials', 'PCAR': 'Industrials',
    'CMI': 'Industrials', 'PH': 'Industrials', 'EMR': 'Industrials',
    'ETN': 'Industrials', 'ROK': 'Industrials', 'AME': 'Industrials',
    'GWW': 'Industrials', 'FAST': 'Industrials', 'MSC': 'Industrials',
    'TT': 'Industrials', 'CARR': 'Industrials', 'OTIS': 'Industrials',
    'JCI': 'Industrials', 'IR': 'Industrials', 'XYL': 'Industrials',
    'XYLD': 'Industrials', 'VLTO': 'Industrials', 'ROP': 'Industrials',
    'ILMN': 'Industrials', 'FDX': 'Industrials', 'UPS': 'Industrials',
    'XPO': 'Industrials', 'SAIA': 'Industrials', 'ODFL': 'Industrials',
    'JBHT': 'Industrials', 'CHRW': 'Industrials', 'EXPD': 'Industrials',
    'URI': 'Industrials', 'RSG': 'Industrials', 'WM': 'Industrials',
    'CTAS': 'Industrials', 'CPRT': 'Industrials', 'VRSK': 'Industrials',
    'HII': 'Industrials', 'LHX': 'Industrials', 'LDOS': 'Industrials',
    'SAIC': 'Industrials', 'BAH': 'Industrials', 'ALLE': 'Industrials',
    'RRX': 'Industrials', 'EME': 'Industrials', 'J': 'Industrials',
    'PWR': 'Industrials', 'MAS': 'Industrials', 'SWK': 'Industrials',

    // â”€â”€ Energy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'EOG': 'Energy', 'SLB': 'Energy', 'HAL': 'Energy',
    'BKR': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'MPC': 'Energy', 'PBF': 'Energy', 'DVN': 'Energy',
    'FANG': 'Energy', 'OXY': 'Energy', 'HES': 'Energy',
    'APA': 'Energy', 'MRO': 'Energy', 'EQT': 'Energy',
    'AR': 'Energy', 'CTRA': 'Energy', 'RRC': 'Energy',
    'KMI': 'Energy', 'WMB': 'Energy', 'OKE': 'Energy',
    'TRGP': 'Energy', 'ET': 'Energy', 'EPD': 'Energy',
    'LNG': 'Energy', 'CQP': 'Energy',

    // â”€â”€ Utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'PCG': 'Utilities',
    'XEL': 'Utilities', 'ED': 'Utilities', 'WEC': 'Utilities',
    'ETR': 'Utilities', 'FE': 'Utilities', 'PPL': 'Utilities',
    'CMS': 'Utilities', 'NI': 'Utilities', 'ATO': 'Utilities',
    'LNT': 'Utilities', 'EVRG': 'Utilities', 'PNW': 'Utilities',
    'VST': 'Utilities', 'NRG': 'Utilities', 'CEG': 'Utilities',
    'FSLR': 'Utilities', 'AES': 'Utilities', 'AWK': 'Utilities',
    'SRE': 'Utilities', 'D': 'Utilities', 'EIX': 'Utilities',
    'ES': 'Utilities', 'CNP': 'Utilities',

    // â”€â”€ Materials â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    'ECL': 'Materials', 'PPG': 'Materials', 'EMN': 'Materials',
    'DD': 'Materials', 'DOW': 'Materials', 'LYB': 'Materials',
    'CE': 'Materials', 'HUN': 'Materials', 'RPM': 'Materials',
    'IFF': 'Materials', 'ALB': 'Materials', 'FCX': 'Materials',
    'NEM': 'Materials', 'GOLD': 'Materials', 'AA': 'Materials',
    'X': 'Materials', 'NUE': 'Materials', 'STLD': 'Materials',
    'RS': 'Materials', 'VMC': 'Materials', 'MLM': 'Materials',
    'PKG': 'Materials', 'IP': 'Materials', 'WRK': 'Materials',
    'CF': 'Materials', 'MOS': 'Materials', 'CTVA': 'Materials',
    'FMC': 'Materials', 'GLD': 'Materials', 'AMCR': 'Materials',
    'MP': 'Materials', 'BALL': 'Materials', 'SON': 'Materials',
    'ANET': 'Materials', 'VLTO': 'Materials',

    // â”€â”€ Real Estate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate',
    'EQIX': 'Real Estate', 'DLR': 'Real Estate', 'SPG': 'Real Estate',
    'O': 'Real Estate', 'PSA': 'Real Estate', 'EQR': 'Real Estate',
    'AVB': 'Real Estate', 'VTR': 'Real Estate', 'VICI': 'Real Estate',
    'WY': 'Real Estate', 'ARE': 'Real Estate', 'BXP': 'Real Estate',
    'KIM': 'Real Estate', 'REG': 'Real Estate', 'FRT': 'Real Estate',
    'NNN': 'Real Estate', 'WELL': 'Real Estate', 'CBRE': 'Real Estate',
    'CSGP': 'Real Estate', 'SBAC': 'Real Estate', 'IRM': 'Real Estate',
    'ESS': 'Real Estate', 'UDR': 'Real Estate', 'CPT': 'Real Estate',
    'MAA': 'Real Estate', 'AIV': 'Real Estate', 'NLY': 'Real Estate',
};

// Group positions by sector using the S&P 500 map, falling back to Finnhub for unknowns
async function groupBySector(positions) {
    const sectors = {};
    const uncached = [];

    // Load any previously Finnhub-fetched overrides from localStorage
    let localCache = {};
    try { localCache = JSON.parse(localStorage.getItem('sector_cache') || '{}'); } catch {}

    positions.forEach(pos => {
        // Priority: localStorage override â†’ S&P 500 map â†’ needs Finnhub lookup
        const sector = localCache[pos.symbol] || SP500_SECTOR_MAP[pos.symbol];
        if (sector) {
            if (!sectors[sector]) sectors[sector] = [];
            sectors[sector].push(pos);
        } else {
            uncached.push(pos);
        }
    });

    // For anything not in the map, try Finnhub as a last resort
    if (uncached.length > 0) {
        console.log(`${uncached.length} tickers not in S&P 500 map, fetching from Finnhub...`);
        const results = await Promise.all(
            uncached.map((pos, i) => fetchSectorFromFinnhub(pos.symbol, i * 200))
        );
        uncached.forEach((pos, i) => {
            const sectorName = results[i] || 'Other';
            localCache[pos.symbol] = sectorName;
            if (!sectors[sectorName]) sectors[sectorName] = [];
            sectors[sectorName].push(pos);
        });
        localStorage.setItem('sector_cache', JSON.stringify(localCache));
    }

    return sectors;
}

// Finnhub fallback for tickers not in the S&P 500 map
async function fetchSectorFromFinnhub(symbol, delayMs = 0) {
    if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
    try {
        const response = await fetch(
            `https://finnhub.io/api/v1/stock/profile2?symbol=${symbol}&token=${FINNHUB_API_KEY}`
        );
        if (!response.ok) return 'Other';
        const data = await response.json();
        const i = (data.finnhubIndustry || '').toLowerCase();
        if (i.includes('technology') || i.includes('semiconductor') || i.includes('software') || i.includes('hardware') || i.includes('electronic')) return 'Technology';
        if (i.includes('financial') || i.includes('bank') || i.includes('insurance') || i.includes('asset management') || i.includes('capital markets')) return 'Financials';
        if (i.includes('health') || i.includes('pharma') || i.includes('biotech') || i.includes('medical')) return 'Health Care';
        if (i.includes('communication') || i.includes('media') || i.includes('entertainment') || i.includes('telecom')) return 'Communication Services';
        if (i.includes('consumer discretionary') || i.includes('retail') || i.includes('restaurant') || i.includes('airline') || i.includes('travel')) return 'Consumer Discretionary';
        if (i.includes('consumer staples') || i.includes('food') || i.includes('beverage') || i.includes('tobacco')) return 'Consumer Staples';
        if (i.includes('industrial') || i.includes('aerospace') || i.includes('defense') || i.includes('machinery') || i.includes('transportation')) return 'Industrials';
        if (i.includes('energy') || i.includes('oil') || i.includes('gas') || i.includes('petroleum')) return 'Energy';
        if (i.includes('utilities') || i.includes('electric') || i.includes('water')) return 'Utilities';
        if (i.includes('real estate') || i.includes('reit') || i.includes('property')) return 'Real Estate';
        if (i.includes('material') || i.includes('chemical') || i.includes('mining') || i.includes('metal') || i.includes('gold')) return 'Materials';
        return 'Other';
    } catch { return 'Other'; }
}

function populateSectorsList(sectors) {
    const totalPortfolioValue = allPositions.reduce((sum, pos) => sum + parseFloat(pos.market_value), 0);

    let sectorsArray = Object.keys(sectors).map(sectorName => {
        const holdings = sectors[sectorName];
        const sectorValue = holdings.reduce((sum, pos) => sum + parseFloat(pos.market_value), 0);
        const allocation = (sectorValue / totalPortfolioValue) * 100;
        const avgReturn = holdings.reduce((sum, pos) => sum + parseFloat(pos.unrealized_plpc || 0), 0) / holdings.length * 100;

        return {
            name: sectorName,
            holdings: holdings,
            value: sectorValue,
            allocation: allocation,
            return: avgReturn,
            count: holdings.length
        };
    });

    sortSectorsArray(sectorsArray, currentSortBy);

    let html = '';
    sectorsArray.forEach((sector, index) => {
        const returnClass = sector.return >= 0 ? 'positive' : 'negative';
        html += `
            <div class="sector-item" onclick="toggleSectorDropdown(${index})">
                <div class="sector-header-row">
                    <h4 class="sector-name">${sector.name}</h4>
                    <div class="sector-stats">
                        <div class="sector-stat">
                            <span class="sector-stat-label">Stocks</span>
                            <span class="sector-stat-value">${sector.count}</span>
                        </div>
                        <div class="sector-stat">
                            <span class="sector-stat-label">Value</span>
                            <span class="sector-stat-value">$${sector.value.toLocaleString('en-US', {minimumFractionDigits: 0})}</span>
                        </div>
                        <div class="sector-stat">
                            <span class="sector-stat-label">Allocation</span>
                            <span class="sector-stat-value">${sector.allocation.toFixed(1)}%</span>
                        </div>
                        <div class="sector-stat">
                            <span class="sector-stat-label">Return</span>
                            <span class="sector-stat-value ${returnClass}">${sector.return >= 0 ? '+' : ''}${sector.return.toFixed(2)}%</span>
                        </div>
                    </div>
                </div>
                <div class="sector-bar">
                    <div class="sector-bar-fill" style="width: ${Math.abs(sector.allocation)}%; background: ${sector.allocation < 0 ? 'var(--negative, #ef4444)' : ''}"></div>
                </div>
                <div class="sector-holdings-dropdown" id="sector-dropdown-${index}">
                    ${sector.holdings.map(pos => {
                        const plpc = parseFloat(pos.unrealized_plpc || 0) * 100;
                        const plClass = plpc >= 0 ? 'positive' : 'negative';
                        return `
                            <div class="sector-holding-item">
                                <div>
                                    <span style="font-weight: 600;">${pos.symbol}</span>
                                    <span style="color: var(--text-secondary); font-size: 13px; margin-left: 8px;">
                                        ${parseFloat(pos.qty)} shares
                                    </span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-weight: 600;">$${parseFloat(pos.market_value).toLocaleString('en-US', {minimumFractionDigits: 2})}</div>
                                    <div class="${plClass}" style="font-size: 13px;">
                                        ${plpc >= 0 ? '+' : ''}${plpc.toFixed(2)}%
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    });

    document.getElementById('sector-list').innerHTML = html;
}

function sortSectorsArray(sectorsArray, sortBy) {
    if (sortBy === 'allocation') {
        sectorsArray.sort((a, b) => b.allocation - a.allocation);
    } else if (sortBy === 'return') {
        sectorsArray.sort((a, b) => b.return - a.return);
    }
}

function sortSectors(sortBy) {
    currentSortBy = sortBy;

    document.querySelectorAll('.sort-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    populateSectorsList(allSectors);
}

function toggleSectorDropdown(index) {
    const dropdown = document.getElementById(`sector-dropdown-${index}`);
    dropdown.classList.toggle('open');
}

// ===== PIE CHART =====
function createSectorPieChart() {
    const ctx = document.getElementById('sectorPieChart');

    if (sectorChart) {
        sectorChart.destroy();
    }

    const totalPortfolioValue = allPositions.reduce((sum, pos) => sum + parseFloat(pos.market_value), 0);

    const labels = [];
    const data = [];
    const colors = [
        '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981',
        '#06b6d4', '#6366f1', '#f97316', '#14b8a6', '#a855f7'
    ];

    Object.keys(allSectors).forEach((sectorName, index) => {
        const holdings = allSectors[sectorName];
        const sectorValue = holdings.reduce((sum, pos) => sum + parseFloat(pos.market_value), 0);
        const allocation = (sectorValue / totalPortfolioValue) * 100;

        labels.push(sectorName);
        data.push(allocation);
    });

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#94a3b8' : '#64748b';

    sectorChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: isDark ? '#0f0f16' : '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cursor: 'pointer',
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const clickedLabel = labels[elements[0].index];
                    navigateToSector(clickedLabel);
                }
            },
            onHover: (event, elements) => {
                event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
            },
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: textColor,
                        padding: 15,
                        font: {
                            size: 12,
                            weight: 600
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Navigate to sectors tab and open the clicked sector's dropdown
function navigateToSector(sectorName) {
    // Switch to sectors tab
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.textContent.trim().includes('Sectors')) item.classList.add('active');
    });
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('sectors-tab').classList.add('active');
    document.getElementById('page-title').textContent = 'Sectors';

    // Find and open the matching sector dropdown
    const sectorItems = document.querySelectorAll('.sector-item');
    sectorItems.forEach((item, index) => {
        const name = item.querySelector('.sector-name').textContent.trim();
        if (name === sectorName) {
            const dropdown = document.getElementById(`sector-dropdown-${index}`);
            if (dropdown && !dropdown.classList.contains('open')) {
                dropdown.classList.add('open');
            }
            // Scroll it into view smoothly
            setTimeout(() => item.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
        }
    });
}

// ===== PERFORMANCE LINE CHART =====
function changeTimePeriod(period) {
    currentTimePeriod = period;

    // Update button states
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update chart with new period
    createPerformanceLineChart();
}

async function createPerformanceLineChart() {
    const ctx = document.getElementById('performanceLineChart');

    if (performanceChart) {
        performanceChart.destroy();
    }

    // Fetch real portfolio history from Alpaca
    const data = await fetchPortfolioHistory(currentTimePeriod);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#94a3b8' : '#64748b';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : '#e2e8f0';

    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [{
                label: 'Portfolio Value',
                data: data.values,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#3b82f6',
                pointHoverBorderColor: '#ffffff',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: isDark ? '#1a1a24' : '#ffffff',
                    titleColor: textColor,
                    bodyColor: textColor,
                    borderColor: gridColor,
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return '$' + context.parsed.y.toLocaleString('en-US', {minimumFractionDigits: 2});
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: textColor,
                        maxTicksLimit: 8
                    }
                },
                y: {
                    grid: {
                        color: gridColor,
                        drawBorder: false
                    },
                    ticks: {
                        color: textColor,
                        callback: function(value) {
                            return '$' + value.toLocaleString('en-US', {minimumFractionDigits: 0});
                        }
                    }
                }
            }
        }
    });
}

async function fetchPortfolioHistory(period) {
    const apiKey = sessionStorage.getItem('alpaca_key');
    const apiSecret = sessionStorage.getItem('alpaca_secret');

     // Added custom range handlers, since Alpaca doesn't support 5Y or ITD directly
    if (period === '5Y') {
        return fetchCustomRangeHistory(apiKey, apiSecret, 5);
    }
    if (period === 'ITD') {
        return fetchSinceInceptionHistory(apiKey, apiSecret);
    }
    // Map period to Alpaca timeframe and period
    const periodMapping = {
        '1D': { timeframe: '5Min', period: '1D' },
        '1W': { timeframe: '1H', period: '1W' },
        '1M': { timeframe: '1D', period: '1M' },
        '3M': { timeframe: '1D', period: '3M' },
        'YTD': { timeframe: '1D', period: '1A' }, // Use 1 year, will filter to YTD
        '5Y': { customRange: '5Y' },   // Handled separately since Alpaca doesn't support 5Y directly
        'ITD': { customRange: 'ITD' }  // Handled separately since Alpaca doesn't support ITD directly
    };

    const config = periodMapping[period] || periodMapping['3M'];

    try {
        const response = await fetch(
            `https://paper-api.alpaca.markets/v2/account/portfolio/history?timeframe=${config.timeframe}&period=${config.period}`,
            {
                headers: {
                    'APCA-API-KEY-ID': apiKey,
                    'APCA-API-SECRET-KEY': apiSecret
                }
            }
        );

        const history = await response.json();

        if (!history.timestamp || !history.equity) {
            console.error('Invalid portfolio history data:', history);
            return generateFallbackData(period);
        }

        // Process the data
        let labels = [];
        let values = [];

        for (let i = 0; i < history.timestamp.length; i++) {
            const timestamp = history.timestamp[i];
            const equity = history.equity[i];

            const date = new Date(timestamp * 1000); // Convert Unix timestamp to Date

            // Format label based on period
            let label;
            if (period === '1D') {
                label = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            } else if (period === '1W') {
                // Hourly points — show weekday + date so each day is clearly distinct on x-axis
                label = date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
            } else if (period === '1M' || period === '3M') {
                label = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            } else {
                label = date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
            }

            labels.push(label);
            values.push(equity);
        }

        // Filter for YTD
        if (period === 'YTD') {
            const yearStart = new Date(new Date().getFullYear(), 0, 1).getTime() / 1000;
            const filteredData = { labels: [], values: [] };

            for (let i = 0; i < history.timestamp.length; i++) {
                if (history.timestamp[i] >= yearStart) {
                    const date = new Date(history.timestamp[i] * 1000);
                    filteredData.labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
                    filteredData.values.push(history.equity[i]);
                }
            }

            return filteredData;
        }

        return { labels, values };

    } catch (error) {
        console.error('Error fetching portfolio history:', error);
        return generateFallbackData(period);
    }
}

// Fetch history for custom range (e.g., 5 years)
async function fetchCustomRangeHistory(apiKey, apiSecret, yearsBack) {
    // Because Alpaca does not support "period=5A",
    // we manually compute the start date (now - 5 years)
    // and fetch daily historical equity values
    const end = new Date();
    const start = new Date();
    start.setFullYear(end.getFullYear() - yearsBack);

    const url =
        `https://paper-api.alpaca.markets/v2/account/portfolio/history?` +
        `start=${start.toISOString().split("T")[0]}` +
        `&end=${end.toISOString().split("T")[0]}` +
        `&timeframe=1D`;

    return fetchAndFormatHistory(url, apiKey, apiSecret);
}

// Fetch history since account inception
async function fetchSinceInceptionHistory(apiKey, apiSecret) {
    // We first fetch the account metadata from Alpaca
    // because this is the ONLY way to know when the
    // account was created.

    // Alpaca will only return equity history FROM the
    // first activity date â€” so inception charts often
    // look "short" for new paper accounts.

    // Fetch account metadata for creation date
    const accountResp = await fetch(
        'https://paper-api.alpaca.markets/v2/account',
        {
            headers: {
                'APCA-API-KEY-ID': apiKey,
                'APCA-API-SECRET-KEY': apiSecret
            }
        }
    );

    const account = await accountResp.json();

    // If Alpaca returns no created_at, bail out safely
    if (!account.created_at) {
        console.warn("Missing created_at, using fallback for ITD.");
        return generateFallbackData('ITD');
    }

    const start = new Date(account.created_at);
    const end = new Date();

    // If the account is extremely new (< 5 days), create a smooth fake history
    const days = (end - start) / 86400000;
    if (days < 5) {
        console.warn("Account too new, generating smooth ITD instead.");
        return generateFallbackData('ITD');
    }

    const url =
        `https://paper-api.alpaca.markets/v2/account/portfolio/history?` +
        `start=${start.toISOString().split("T")[0]}` +
        `&end=${end.toISOString().split("T")[0]}` +
        `&timeframe=1D`;

    return fetchAndFormatHistory(url, apiKey, apiSecret);
}

// Helper to fetch and format history data
async function fetchAndFormatHistory(url, apiKey, apiSecret) {
    // Converts Alpaca timestamps â†’ chart labels.
    //
    // This is used for BOTH:
    //   - Custom 5Y history
    //   - Inception-to-date history

    try {
        const response = await fetch(url, {
            headers: {
                'APCA-API-KEY-ID': apiKey,
                'APCA-API-SECRET-KEY': apiSecret
            }
        });

        const history = await response.json();

        if (!history.timestamp || !history.equity) {
            console.error("Invalid Alpaca history:", history);
            return generateFallbackData('5Y');
        }

        let labels = [];
        let values = [];

        for (let i = 0; i < history.timestamp.length; i++) {
            const date = new Date(history.timestamp[i] * 1000);
            labels.push(
                date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
            );
            values.push(history.equity[i]);
        }

        return { labels, values };

    } catch (err) {
        console.error("History fetch error:", err);
        return generateFallbackData('5Y');
    }
}

function generateFallbackData(period) {
    // Fallback to simulated data if API fails
    const now = new Date();
    let dataPoints = [];
    let labels = [];
    let currentValue = 100000;

    const periodConfig = {
        '1D': { points: 78, interval: 5, unit: 'minutes' },
        '1W': { points: 5, interval: 1, unit: 'days' },
        '1M': { points: 21, interval: 1, unit: 'days' },
        '3M': { points: 63, interval: 1, unit: 'days' },
        'YTD': { points: new Date().getMonth() * 21, interval: 1, unit: 'days' },
        '5Y': { points: 60, interval: 1, unit: 'months' },
        'ITD': { points: 100, interval: 1, unit: 'weeks' }
    };

    const config = periodConfig[period] || periodConfig['3M'];

    for (let i = 0; i < config.points; i++) {
        const date = new Date(now);

        if (config.unit === 'minutes') {
            date.setMinutes(date.getMinutes() - (config.points - i) * config.interval);
            labels.push(date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
        } else if (config.unit === 'days') {
            date.setDate(date.getDate() - (config.points - i) * config.interval);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        } else if (config.unit === 'months') {
            date.setMonth(date.getMonth() - (config.points - i) * config.interval);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }));
        } else if (config.unit === 'weeks') {
            date.setDate(date.getDate() - (config.points - i) * 7);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }));
        }

        const trend = 0.0003;
        const randomChange = (Math.random() - 0.5) * 0.01;
        currentValue = currentValue * (1 + trend + randomChange);
        dataPoints.push(currentValue);
    }

    return { labels, values: dataPoints };
}
// ===== EMAIL SUBSCRIPTION =====
async function subscribeEmail() {
    const emailInput = document.getElementById('subscribe-email');
    const email = emailInput.value.trim();
    const messageDiv = document.getElementById('subscription-message');

    if (!email) {
        messageDiv.style.display = 'block';
        messageDiv.style.color = 'var(--negative)';
        messageDiv.textContent = 'âš ï¸ Please enter a valid email address';
        return;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        messageDiv.style.display = 'block';
        messageDiv.style.color = 'var(--negative)';
        messageDiv.textContent = 'âš ï¸ Please enter a valid email address';
        return;
    }

    try {
        // Read current subscribers
        let subscribers = ['andrew.khan921@gmail.com']; // Default

        // Add new subscriber if not already present
        if (!subscribers.includes(email)) {
            subscribers.push(email);
        }

        // Save to localStorage and file
        const subscriberData = { subscribers: subscribers };
        localStorage.setItem('email_subscribers', JSON.stringify(subscriberData));

        // Show success message
        messageDiv.style.display = 'block';
        messageDiv.style.color = 'var(--positive)';
        messageDiv.innerHTML = 'âœ… Successfully subscribed! You\'ll receive notifications at ' + email;

        // Also save to subscribers.json file (in production, this would be a backend API call)
        console.log('Subscriber added:', email);
        console.log('Save this email to subscribers.json manually or via backend API');

        // Clear input
        emailInput.value = '';

        // Hide message after 5 seconds
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, 5000);

    } catch (error) {
        console.error('Error subscribing:', error);
        messageDiv.style.display = 'block';
        messageDiv.style.color = 'var(--negative)';
        messageDiv.textContent = 'âš ï¸ Error subscribing. Please try again.';
    }
}

// ===== NEWS FEED =====
let newsCache = null;
let newsCacheTime = null;

async function loadNewsFeed() {
    const newsFeed = document.getElementById('news-feed');

    // Serve cache if less than 10 minutes old (prevents burning API limits on every tab switch)
    if (newsCache && newsCacheTime && (Date.now() - newsCacheTime) < 10 * 60 * 1000) {
        console.log('Serving news from cache');
        renderNewsCards(newsCache);
        return;
    }

    newsFeed.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--text-secondary);"><i class="fas fa-spinner fa-spin"></i> Loading news...</div>';

    const results = await Promise.allSettled([

        // --- Alpaca ---
        (async () => {
            const apiKey = sessionStorage.getItem('alpaca_key');
            const apiSecret = sessionStorage.getItem('alpaca_secret');
            const response = await fetch('https://data.alpaca.markets/v1beta1/news?limit=10&sort=desc', {
                headers: { 'APCA-API-KEY-ID': apiKey, 'APCA-API-SECRET-KEY': apiSecret }
            });
            if (!response.ok) throw new Error(`Alpaca: ${response.status}`);
            const data = await response.json();
            console.log(`Alpaca: ${(data.news || []).length} articles`);
            return (data.news || []).map(a => ({
                source: a.source || 'Market News',
                title: a.headline,
                summary: a.summary || a.content?.substring(0, 150) + '...' || '',
                url: a.url,
                timestamp: new Date(a.created_at).getTime(),
                time: getTimeAgo(new Date(a.created_at)),
                tags: (a.symbols || []).slice(0, 3),
                sentiment: analyzeSentiment(a.headline)
            }));
        })(),

        // --- Finnhub ---
        (async () => {
            const response = await fetch(`https://finnhub.io/api/v1/news?category=general&token=${FINNHUB_API_KEY}`);
            if (!response.ok) throw new Error(`Finnhub: ${response.status}`);
            const data = await response.json();
            console.log(`Finnhub: ${(data || []).length} articles`);
            return (data || []).slice(0, 8).map(a => ({
                source: a.source || 'Finnhub',
                title: a.headline,
                summary: a.summary || '',
                url: a.url,
                timestamp: new Date(a.datetime * 1000).getTime(),
                time: getTimeAgo(new Date(a.datetime * 1000)),
                tags: [a.category || 'Market'].filter(Boolean),
                sentiment: analyzeSentiment(a.headline)
            }));
        })(),

        // --- Marketaux ---
        (async () => {
            const response = await fetch(`https://api.marketaux.com/v1/news/all?language=en&limit=6&api_token=${MARKETAUX_API_KEY}`);
            if (!response.ok) throw new Error(`Marketaux: ${response.status}`);
            const data = await response.json();
            console.log(`Marketaux: ${(data.data || []).length} articles`);
            return (data.data || []).map(a => ({
                source: a.source || 'Marketaux',
                title: a.title,
                summary: a.description || '',
                url: a.url,
                timestamp: new Date(a.published_at).getTime(),
                time: getTimeAgo(new Date(a.published_at)),
                tags: (a.entities || []).slice(0, 3).map(e => e.symbol).filter(Boolean),
                sentiment: analyzeSentiment(a.title)
            }));
        })()

    ]);

    const [alpacaArticles, finnhubArticles, marketauxArticles] = results.map((r, i) => {
        if (r.status === 'rejected') {
            console.warn(`News source ${['Alpaca', 'Finnhub', 'Marketaux'][i]} failed:`, r.reason.message);
            return [];
        }
        return r.value;
    });

    const allArticles = [...alpacaArticles, ...finnhubArticles, ...marketauxArticles]
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 20);

    if (allArticles.length === 0) {
        renderNewsCards(generateSampleNews());
        return;
    }

    newsCache = allArticles;
    newsCacheTime = Date.now();
    renderNewsCards(allArticles);
}

function renderNewsCards(articles) {
    const newsFeed = document.getElementById('news-feed');
    let html = '';
    articles.forEach(article => {
        const sentimentClass = article.sentiment;
        const sentimentIcon = article.sentiment === 'bullish' ? 'fa-arrow-trend-up' :
                              article.sentiment === 'bearish' ? 'fa-arrow-trend-down' : 'fa-minus';
        html += `
            <div class="news-card" onclick="window.open('${article.url}', '_blank')">
                <div class="news-source">
                    <div class="news-source-logo"></div>
                    <span class="news-source-name">${article.source}</span>
                    <span class="news-time">${article.time}</span>
                </div>
                <h4 class="news-title">${article.title}</h4>
                <p class="news-summary">${article.summary}</p>
                <div class="news-tags">
                    <span class="news-sentiment ${sentimentClass}">
                        <i class="fas ${sentimentIcon}"></i> ${article.sentiment}
                    </span>
                    ${article.tags.map(tag => `<span class="news-tag">${tag}</span>`).join('')}
                </div>
            </div>
        `;
    });
    newsFeed.innerHTML = html;
}

function analyzeSentiment(headline) {
    // Simple sentiment analysis based on keywords
    const bullishWords = ['rally', 'surge', 'gain', 'jump', 'rise', 'soar', 'beat', 'exceed', 'growth', 'strong', 'positive', 'bullish', 'up'];
    const bearishWords = ['fall', 'drop', 'decline', 'plunge', 'slump', 'loss', 'miss', 'weak', 'negative', 'bearish', 'down', 'crash'];

    const lowerHeadline = headline.toLowerCase();

    const bullishCount = bullishWords.filter(word => lowerHeadline.includes(word)).length;
    const bearishCount = bearishWords.filter(word => lowerHeadline.includes(word)).length;

    if (bullishCount > bearishCount) return 'bullish';
    if (bearishCount > bullishCount) return 'bearish';
    return 'neutral';
}

function getTimeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);

    let interval = seconds / 31536000;
    if (interval > 1) return Math.floor(interval) + ' years ago';

    interval = seconds / 2592000;
    if (interval > 1) return Math.floor(interval) + ' months ago';

    interval = seconds / 86400;
    if (interval > 1) return Math.floor(interval) + ' days ago';

    interval = seconds / 3600;
    if (interval > 1) return Math.floor(interval) + ' hours ago';

    interval = seconds / 60;
    if (interval > 1) return Math.floor(interval) + ' minutes ago';

    return 'Just now';
}

function generateSampleNews() {
    // Sample news data - in production, fetch from real news API
    return [
        {
            source: 'Reuters',
            time: '2 hours ago',
            title: 'Fed Signals Potential Rate Cut as Inflation Cools',
            summary: 'The Federal Reserve indicated it may consider reducing interest rates in the coming months as inflation continues to trend toward its 2% target.',
            sentiment: 'bullish',
            tags: ['Federal Reserve', 'Interest Rates', 'Macro'],
            url: '#'
        },
        {
            source: 'Bloomberg',
            time: '4 hours ago',
            title: 'Tech Sector Rallies on Strong Earnings Reports',
            summary: 'Major technology companies exceeded earnings expectations, driving the Nasdaq to new highs as AI investments continue to pay off.',
            sentiment: 'bullish',
            tags: ['Technology', 'Earnings', 'AI'],
            url: '#'
        },
        {
            source: 'CNBC',
            time: '6 hours ago',
            title: 'Energy Stocks Under Pressure Amid Oil Price Decline',
            summary: 'Crude oil prices fell 3% today on concerns about weakening demand, putting pressure on energy sector equities.',
            sentiment: 'bearish',
            tags: ['Energy', 'Oil', 'Commodities'],
            url: '#'
        },
        {
            source: 'WSJ',
            time: '8 hours ago',
            title: 'Consumer Spending Remains Resilient Despite Economic Headwinds',
            summary: 'Retail sales data showed continued strength in consumer spending, suggesting the economy remains robust.',
            sentiment: 'neutral',
            tags: ['Retail', 'Economy', 'Consumer'],
            url: '#'
        },
        {
            source: 'MarketWatch',
            time: '10 hours ago',
            title: 'Financials Gain Ground on Banking Sector Optimism',
            summary: 'Financial stocks advanced as investors bet on improved lending margins and stronger loan growth.',
            sentiment: 'bullish',
            tags: ['Financials', 'Banking', 'Lending'],
            url: '#'
        },
        {
            source: 'FT',
            time: '12 hours ago',
            title: 'Healthcare Sector Faces Regulatory Uncertainty',
            summary: 'Proposed changes to drug pricing policies have created volatility in pharmaceutical and biotech stocks.',
            sentiment: 'bearish',
            tags: ['Healthcare', 'Pharma', 'Regulation'],
            url: '#'
        }
    ];
}

function refreshNews() {
    newsCache = null;
    newsCacheTime = null;
    loadNewsFeed();
}


// ===== SETTINGS =====
function openSettings() {
    document.getElementById('settings-modal').classList.add('active');

    // Load current API key (masked)
    const apiKey = sessionStorage.getItem('alpaca_key');
    if (apiKey) {
        document.getElementById('current-api-key').value = apiKey.substring(0, 2) + 'â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢';
        document.getElementById('current-secret-key').value = 'â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢';
    }
}

function closeSettings() {
    document.getElementById('settings-modal').classList.remove('active');
}

function saveSettings() {
    const settings = {
        positionSize: document.getElementById('position-size').value,
        rebalanceFreq: document.getElementById('rebalance-freq').value,
        orderType: document.getElementById('order-type').value,
        maxDrawdown: document.getElementById('max-drawdown').value,
        stopLoss: document.getElementById('stop-loss').value,
        takeProfit: document.getElementById('take-profit').value,
        email: document.getElementById('email').value,
        phone: document.getElementById('phone').value,
        alertTrade: document.getElementById('alert-trade').checked,
        alertPerformance: document.getElementById('alert-performance').checked,
        alertDaily: document.getElementById('alert-daily').checked,
        alertPrice: document.getElementById('alert-price').checked,
        notifyMarketOpen: document.getElementById('notify-market-open').checked,
        notifyMidday: document.getElementById('notify-midday').checked,
        notifyMarketClose: document.getElementById('notify-market-close').checked
    };

    localStorage.setItem('trading_settings', JSON.stringify(settings));
    closeSettings();
    alert('Settings saved successfully!');
}

// Load saved settings
function loadSettings() {
    const saved = localStorage.getItem('trading_settings');
    if (saved) {
        const settings = JSON.parse(saved);
        document.getElementById('position-size').value = settings.positionSize || 45;
        document.getElementById('rebalance-freq').value = settings.rebalanceFreq || 'weekly';
        document.getElementById('order-type').value = settings.orderType || 'market';
        document.getElementById('max-drawdown').value = settings.maxDrawdown || 20;
        document.getElementById('stop-loss').value = settings.stopLoss || 10;
        document.getElementById('take-profit').value = settings.takeProfit || 20;
        document.getElementById('email').value = settings.email || '';
        document.getElementById('phone').value = settings.phone || '';
        document.getElementById('alert-trade').checked = settings.alertTrade !== false;
        document.getElementById('alert-performance').checked = settings.alertPerformance !== false;
        document.getElementById('alert-daily').checked = settings.alertDaily || false;
        document.getElementById('alert-price').checked = settings.alertPrice || false;
        document.getElementById('notify-market-open').checked = settings.notifyMarketOpen !== false;
        document.getElementById('notify-midday').checked = settings.notifyMidday || false;
        document.getElementById('notify-market-close').checked = settings.notifyMarketClose !== false;
    }
}

// Close modal on outside click
document.addEventListener('click', function(e) {
    const modal = document.getElementById('settings-modal');
    if (e.target === modal) {
        closeSettings();
    }
});

// ===== INITIALIZATION =====
window.addEventListener('DOMContentLoaded', () => {
    loadSettings();

    const apiKey = sessionStorage.getItem('alpaca_key');
    if (apiKey) {
        document.getElementById('login-page').style.display = 'none';
        document.getElementById('main-dashboard').style.display = 'block';
        loadDashboardData();
    }
});
// ===== QUANT AI CHAT =====
let chatSessions = {
    'chat-session-1': []
};
let activeChatId = 'chat-session-1';
let chatSessionCounter = 1;
const QUANT_AI_URL = 'https://hdo6lukv03.execute-api.us-east-1.amazonaws.com/prod/quant-ai-chat';

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    autoResize(input);
    appendUserMessage(message);

    // Hide welcome screen if visible
    const welcome = document.querySelector('.chat-welcome');
    if (welcome) welcome.style.display = 'none';

    // Save to session
    chatSessions[activeChatId].push({ role: 'user', content: message });

    // Update history label with first message
    const historyItem = document.getElementById(activeChatId);
    if (historyItem && chatSessions[activeChatId].length === 1) {
        historyItem.querySelector('span').textContent = message.substring(0, 28) + (message.length > 28 ? '...' : '');
    }

    // Show typing indicator
    const typingId = showTypingIndicator();
    document.getElementById('chat-send-btn').disabled = true;

    try {
        const response = await fetch(QUANT_AI_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: chatSessions[activeChatId] })
        });

        removeTypingIndicator(typingId);

        const data = await response.json();
        const reply = data.reply;

        chatSessions[activeChatId].push({ role: 'assistant', content: reply });
        appendAssistantMessage(reply);

    } catch (err) {
        removeTypingIndicator(typingId);
        appendAssistantMessage('Sorry, I encountered an error. Please check your connection and try again.');
        console.error('Quant AI error:', err);
    }

    document.getElementById('chat-send-btn').disabled = false;
}

function appendUserMessage(text) {
    const messages = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-message user';
    div.innerHTML = `
        <div class="chat-bubble">${escapeHtml(text)}</div>
        <div class="chat-avatar"><i class="fas fa-user"></i></div>
    `;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function appendAssistantMessage(text) {
    const messages = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-message assistant';
    // Simple markdown-ish: bold, line breaks
    const formatted = escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    div.innerHTML = `
        <div class="chat-avatar"><img src="assets/husky_logo.jpg" alt="AI"></div>
        <div class="chat-bubble">${formatted}</div>
    `;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function showTypingIndicator() {
    const messages = document.getElementById('chat-messages');
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'chat-message assistant';
    div.id = id;
    div.innerHTML = `
        <div class="chat-avatar"><img src="assets/husky_logo.jpg" alt="AI"></div>
        <div class="chat-bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function handleChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    sendMessage();
}

function startNewChat() {
    chatSessionCounter++;
    const id = 'chat-session-' + chatSessionCounter;
    chatSessions[id] = [];

    const list = document.getElementById('chat-history-list');
    const item = document.createElement('div');
    item.className = 'chat-history-item';
    item.id = id;
    item.onclick = () => loadChat(id);
    item.innerHTML = `<i class="fas fa-comment-dots"></i><span>New Conversation</span>`;
    list.insertBefore(item, list.firstChild);

    loadChat(id);
}

function loadChat(id) {
    activeChatId = id;

    // Update active state
    document.querySelectorAll('.chat-history-item').forEach(i => i.classList.remove('active'));
    document.getElementById(id).classList.add('active');

    // Re-render chat
    const messages = document.getElementById('chat-messages');
    messages.innerHTML = '';

    if (chatSessions[id].length === 0) {
        messages.innerHTML = `
            <div class="chat-welcome">
                <div class="chat-welcome-logo"><img src="assets/husky_logo.jpg" alt="UConn Quant AI"></div>
                <h2>Quant AI Assistant</h2>
                <p>Ask me anything about your portfolio, market trends, trading strategies, or financial analysis.</p>
                <div class="chat-suggestions">
                    <button class="suggestion-chip" onclick="sendSuggestion('What is momentum trading and how does it work?')">What is momentum trading?</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('Analyze the current market conditions and key risks to watch.')">Analyze market conditions</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('Explain sector rotation strategy in simple terms.')">Explain sector rotation</button>
                    <button class="suggestion-chip" onclick="sendSuggestion('What are the best risk management techniques for an algorithmic portfolio?')">Risk management tips</button>
                </div>
            </div>`;
    } else {
        chatSessions[id].forEach(msg => {
            if (msg.role === 'user') appendUserMessage(msg.content);
            else appendAssistantMessage(msg.content);
        });
    }
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}