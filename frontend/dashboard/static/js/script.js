let sectorChart = null;
let performanceChart = null;
let currentTimePeriod = '3M';
let currentSortBy = 'allocation';
let allPositions = [];
let allSectors = {};
let performanceData = [];

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

// ===== DATA LOADING =====
async function loadDashboardData() {
  try {
    const apiKey = sessionStorage.getItem('alpaca_key');
    const apiSecret = sessionStorage.getItem('alpaca_secret');

    // --- Fetch account data ---
    const accountResponse = await fetch('/api/account', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey, apiSecret })
    });
    const account = await accountResponse.json();

    // --- Fetch positions data ---
    const positionsResponse = await fetch('/api/positions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey, apiSecret })
    });
    const positions = await positionsResponse.json();

    // Render results
    populateDashboard(account, positions);
    populatePositionsTable(positions);
    allPositions = positions;
    allSectors = groupBySector(positions);
    populateSectorsList(allSectors);
    createSectorPieChart();
    createPerformanceLineChart();


  } catch (error) {
    console.error("Error communicating with Flask:", error);
  }
}

// ===== DASHBOARD =====
function populateDashboard(account, positions) {
    const portfolioValue = parseFloat(account.portfolio_value);
    const initialCapital = 100000;
    const totalReturn = ((portfolioValue - initialCapital) / initialCapital) * 100;
    const dollarGain = portfolioValue - initialCapital;

    const totalUnrealizedPL = positions.reduce((sum, pos) => sum + parseFloat(pos.unrealized_pl || 0), 0);
    const todayChangePC = (totalUnrealizedPL / (portfolioValue - totalUnrealizedPL)) * 100;

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

    // Performance stats
    const statsHTML = `
        <div style="padding: 20px;">
            <div style="margin-bottom: 24px;">
                <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">Weekly Change</p>
                <p style="font-size: 28px; font-weight: 700; margin-bottom: 4px; color: var(--positive);">+$3,247.50</p>
                <p style="font-size: 14px; color: var(--positive);"><i class="fas fa-arrow-up"></i> +3.35%</p>
            </div>
            <div style="margin-bottom: 24px;">
                <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px;">Monthly Change</p>
                <p style="font-size: 28px; font-weight: 700; margin-bottom: 4px; color: var(--positive);">+$2,547.89</p>
                <p style="font-size: 14px; color: var(--positive);"><i class="fas fa-arrow-up"></i> +2.61%</p>
            </div>
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
                            <img src="https://logo.clearbit.com/${getCompanyDomain(pos.symbol)}"
                                    onerror="this.style.display='none'; this.parentElement.innerHTML='${pos.symbol.substring(0, 2)}';">
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

// ===== SECTORS =====
function groupBySector(positions) {
    const sectors = {};
    const sectorMap = {
        'Technology': ['PLTR', 'APP', 'AVGO', 'JBL', 'APH', 'GLW'],
        'Financials': ['HOOD', 'IBKR', 'C', 'COIN', 'SCHW', 'GS'],
        'Utilities': ['VST', 'NRG', 'CEG'],
        'Consumer Discretionary': ['TPR', 'RCL', 'DASH', 'CCL', 'LYV'],
        'Industrials': ['GEV', 'UAL', 'AXON', 'HWM', 'GE', 'JCI', 'EME'],
        'Communication Services': ['NFLX'],
        'Energy': ['EQT', 'BKR'],
        'Health Care': ['GILD', 'MCK', 'CAH', 'PODD', 'CVS'],
        'Consumer Staples': ['MO', 'DLTR', 'PM'],
        'Real Estate': ['WELL', 'CBRE', 'CSGP'],
        'Materials': ['CTVA', 'NEM', 'MOS']
    };

    positions.forEach(pos => {
        let sectorName = 'Other';
        for (const [sector, symbols] of Object.entries(sectorMap)) {
            if (symbols.includes(pos.symbol)) {
                sectorName = sector;
                break;
            }
        }

        if (!sectors[sectorName]) {
            sectors[sectorName] = [];
        }
        sectors[sectorName].push(pos);
    });

    return sectors;
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
                    <div class="sector-bar-fill" style="width: ${sector.allocation}%"></div>
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

    // Map period to Alpaca timeframe and period
    const periodMapping = {
        '1D': { timeframe: '5Min', period: '1D' },
        '1W': { timeframe: '1H', period: '1W' },
        '1M': { timeframe: '1D', period: '1M' },
        '3M': { timeframe: '1D', period: '3M' },
        'YTD': { timeframe: '1D', period: '1A' }, // Use 1 year, will filter to YTD
        '5Y': { timeframe: '1W', period: '5A' },
        'ITD': { timeframe: '1W', period: 'all' }
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
            } else if (period === '1W' || period === '1M' || period === '3M') {
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