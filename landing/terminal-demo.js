// Terminal Demo Animation for Luna 9
// Shows rotating scenarios of CLI usage

class TerminalDemo {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.isPaused = false;
        this.currentScenario = 0;
        this.animationTimeout = null;
        this.currentAnimation = null;

        // Define all scenarios
        this.scenarios = [
            {
                name: "log-analysis",
                prompt: "🌙",
                commands: [
                    {
                        type: "command",
                        text: "$ luna9 show me the logs I think there is a latency issue",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `🌙 I can help with that! Which log file should I check?
   Available: server.log, api.log, instance.log`,
                        delay: 1200
                    },
                    {
                        type: "command",
                        text: "$ cat instance.log | grep ERROR | luna9",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `🌙 Found 3 ERROR entries in the piped data:

   1. Connection timeout at 14:23:45
   2. Database query latency spike (2.3s) at 14:24:12
   3. Redis connection refused at 14:25:01

   The latency issue started at 14:24:12 - looks like DB performance.
   Want me to check related entries?`,
                        delay: 1800
                    }
                ]
            },
            {
                name: "code-search",
                prompt: "[Luna9]",
                commands: [
                    {
                        type: "command",
                        text: "$ luna9 hey, that refactoring from yesterday",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `[Luna9] I see you worked on the authentication refactor yesterday!
        Want me to show:
        1. The files you changed
        2. The main changes made
        3. Related TODO comments

        Which would help?`,
                        delay: 1400
                    },
                    {
                        type: "command",
                        text: "$ luna9 show me the main changes and any TODOs",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `[Luna9] Main changes in auth refactor:
        • Moved JWT validation to middleware (auth.js:45)
        • Extracted token refresh logic (tokens.js:23)
        • Added rate limiting (middleware/rateLimit.js)

        Found 2 related TODOs:
        • auth.js:78 - "TODO: Add refresh token rotation"
        • tokens.js:91 - "TODO: Consider Redis for token blacklist"`,
                        delay: 1600
                    }
                ]
            },
            {
                name: "git-workflow",
                prompt: "L9>",
                commands: [
                    {
                        type: "command",
                        text: "$ git diff | luna9 write me a commit message",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `L9> Analyzing your changes...

    Suggested commit message:

    feat: add user session persistence

    - Implement Redis-backed session store
    - Add session cleanup cron job
    - Update user logout to clear sessions
    - Add tests for session expiration

    Want me to commit this, or would you like to edit?`,
                        delay: 1500
                    },
                    {
                        type: "command",
                        text: "$ luna9 looks good, commit it",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `L9> ✓ Committed with message above
    [main 7a3f2e1] feat: add user session persistence
     4 files changed, 127 insertions(+), 12 deletions(-)`,
                        delay: 600
                    }
                ]
            },
            {
                name: "container-debug",
                prompt: "luna:",
                commands: [
                    {
                        type: "command",
                        text: "$ docker logs api-container --tail 100 | luna9",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `luna: Analyzing container logs...

      Detected issues:
      1. Memory warnings (3x) - container approaching 512MB limit
      2. Connection pool exhausted - suggests high load
      3. Slow query logged at 03:14:22 (8.2s)

      Recommendation: Increase memory limit to 1GB and investigate
      the slow query in users table.`,
                        delay: 1500
                    },
                    {
                        type: "command",
                        text: "$ luna9 show me that slow query",
                        delay: 100
                    },
                    {
                        type: "response",
                        text: `luna: Here's the query from the logs:

      SELECT * FROM users
      LEFT JOIN user_preferences ON users.id = user_preferences.user_id
      WHERE users.created_at > '2024-01-01'

      Missing index on users.created_at causing full table scan.
      Run: CREATE INDEX idx_users_created ON users(created_at);`,
                        delay: 1400
                    }
                ]
            }
        ];

        this.init();
    }

    init() {
        this.render();
        this.startAnimation();
    }

    render() {
        const scenario = this.scenarios[this.currentScenario];

        this.container.innerHTML = `
            <div class="terminal-window">
                <div class="terminal-header">
                    <div class="terminal-controls">
                        <span class="control close"></span>
                        <span class="control minimize"></span>
                        <span class="control maximize"></span>
                    </div>
                    <div class="terminal-title">luna9 - ${scenario.name}</div>
                    <div class="terminal-actions">
                        <button class="pause-btn" id="pause-demo">
                            ${this.isPaused ? '▶' : '⏸'}
                        </button>
                    </div>
                </div>
                <div class="terminal-body" id="terminal-content">
                    <div class="terminal-line">
                        <span class="cursor">█</span>
                    </div>
                </div>
                <div class="prompt-info" data-prompt="${scenario.prompt}">
                    Hover over prompt to see customization options
                </div>
            </div>
        `;

        // Add pause button listener
        document.getElementById('pause-demo').addEventListener('click', () => {
            this.togglePause();
        });

        // Add hover tooltip
        const promptInfo = this.container.querySelector('.prompt-info');
        promptInfo.addEventListener('mouseenter', () => {
            this.showPromptTooltip(promptInfo, scenario.prompt);
        });
    }

    showPromptTooltip(element, prompt) {
        const tooltip = document.createElement('div');
        tooltip.className = 'prompt-tooltip';
        tooltip.innerHTML = `
            <strong>Customize your prompt!</strong>
            <p>Current: <code>${prompt}</code></p>
            <p>Options: 🌙 [Luna9] L9> luna: or anything you like</p>
        `;
        element.appendChild(tooltip);

        element.addEventListener('mouseleave', () => {
            tooltip.remove();
        }, { once: true });
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        const btn = document.getElementById('pause-demo');
        btn.textContent = this.isPaused ? '▶' : '⏸';

        if (this.isPaused) {
            // Clear any pending timeouts when pausing
            if (this.animationTimeout) {
                clearTimeout(this.animationTimeout);
                this.animationTimeout = null;
            }
        }
    }

    async startAnimation() {
        await this.animateScenario(this.currentScenario);

        if (!this.isPaused) {
            // Wait before next scenario - longer delay for reading
            this.animationTimeout = setTimeout(() => {
                this.currentScenario = (this.currentScenario + 1) % this.scenarios.length;
                this.render();
                this.startAnimation();
            }, 5000);
        }
    }

    async animateScenario(index) {
        const scenario = this.scenarios[index];
        const content = document.getElementById('terminal-content');

        for (const command of scenario.commands) {
            if (this.isPaused) {
                await this.waitForUnpause();
            }

            await this.typeText(content, command.text, command.type, command.delay);
            await this.sleep(command.delay);
        }
    }

    async typeText(container, text, type, delayAfter) {
        // Remove cursor from last line
        const lastCursor = container.querySelector('.cursor');
        if (lastCursor) lastCursor.remove();

        const line = document.createElement('div');
        line.className = `terminal-line ${type}`;

        // For commands, add them instantly with fade
        if (type === 'command') {
            line.textContent = text;
            line.style.opacity = '0';
            container.appendChild(line);

            await this.sleep(50);
            line.style.transition = 'opacity 0.3s';
            line.style.opacity = '1';
        } else {
            // For responses, show with fade
            line.textContent = text;
            line.style.opacity = '0';
            container.appendChild(line);

            await this.sleep(100);
            line.style.transition = 'opacity 0.5s';
            line.style.opacity = '1';
        }

        // Add cursor to new line
        const cursorLine = document.createElement('div');
        cursorLine.className = 'terminal-line';
        cursorLine.innerHTML = '<span class="cursor">█</span>';
        container.appendChild(cursorLine);

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async waitForUnpause() {
        return new Promise(resolve => {
            const check = setInterval(() => {
                if (!this.isPaused) {
                    clearInterval(check);
                    resolve();
                }
            }, 100);
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new TerminalDemo('terminal-demo-container');
});
