// Desktop Mockup Component for Luna 9
// Shows two collaboration scenarios: Learning & Creative (RPG)

class DesktopMockup {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentScenario = 'creative'; // Start with creative (RPG) to showcase values
        this.currentTheme = 'dark';

        this.scenarios = {
            learning: {
                title: "Learning: Byzantine Fault Tolerance",
                goals: [
                    "Explore consensus mechanisms",
                    "Map to real-world use cases",
                    "Understand trade-offs"
                ],
                folders: [
                    { name: "research", files: ["consensus.md", "papers.md"] },
                    { name: "notes", files: ["bft-overview.md", "questions.md"] }
                ],
                graph: [
                    { label: "BFT Basics", active: false },
                    { label: "Consensus", active: true },
                    { label: "Trade-offs", active: false },
                    { label: "Applications", active: false }
                ],
                messages: [
                    {
                        type: "user",
                        text: "Can you help me understand Byzantine fault tolerance trade-offs in distributed systems?"
                    },
                    {
                        type: "assistant",
                        text: "Sure! What aspects interest you most? Performance, security, or specific use cases?"
                    },
                    {
                        type: "user",
                        text: "Why don't all systems use BFT if it's more resilient?"
                    }
                ]
            },
            creative: {
                title: "Creative: Tabletop RPG Development",
                goals: [
                    "Finish character art for Chapter 3",
                    "Verify combat mechanics balance",
                    "Integrate story chapters into rulebook"
                ],
                folders: [
                    { name: "art", files: ["characters.psd", "items.psd"] },
                    { name: "mechanics", files: ["stats.md", "combat.md"] },
                    { name: "story", files: ["chapter1.md", "chapter2.md", "chapter3.md"] }
                ],
                graph: [
                    { label: "Art Assets", active: true },
                    { label: "Game Mechanics", active: true },
                    { label: "Story Integration", active: false },
                    { label: "Rulebook Draft", active: false }
                ],
                messages: [
                    {
                        type: "user",
                        text: "I finished the character art for Chapter 3! Can you help me check if the stat progression math works at high levels?"
                    },
                    {
                        type: "assistant",
                        text: "Congrats on the art! Let me check the formula you wrote: BaseValue + (Level × Modifier) + (Level² × 0.1). Running the numbers at L20 and L30..."
                    },
                    {
                        type: "user",
                        text: "Perfect - does it get too powerful after level 20?"
                    }
                ]
            }
        };

        this.init();
    }

    init() {
        this.render();
        this.attachEventListeners();
    }

    render() {
        const scenario = this.scenarios[this.currentScenario];
        const themeClass = this.currentTheme === 'dark' ? '' : 'light-theme';

        this.container.innerHTML = `
            <div class="mockup-controls">
                <button class="mockup-btn ${this.currentScenario === 'learning' ? 'active' : ''}" data-scenario="learning">
                    Learning Scenario
                </button>
                <button class="mockup-btn ${this.currentScenario === 'creative' ? 'active' : ''}" data-scenario="creative">
                    Creative Scenario (RPG)
                </button>
                <button class="theme-toggle" id="theme-toggle">
                    <span class="theme-icon">${this.currentTheme === 'dark' ? '☀️' : '🌙'}</span>
                    <span>${this.currentTheme === 'dark' ? 'Light' : 'Dark'} Mode</span>
                </button>
            </div>

            <div class="desktop-app ${themeClass}">
                <div class="app-header">
                    <div class="window-controls">
                        <span class="window-control close"></span>
                        <span class="window-control minimize"></span>
                        <span class="window-control maximize"></span>
                    </div>
                    <div class="app-title">
                        🌙 Luna9 - ${scenario.title}
                    </div>
                    <div style="width: 60px;"></div>
                </div>

                <div class="app-body">
                    <!-- Left Sidebar: Files & Goals -->
                    <div class="left-sidebar">
                        <div class="sidebar-section">
                            <div class="section-title">Project Files</div>
                            <div class="folder-tree">
                                ${this.renderFolders(scenario.folders)}
                            </div>
                        </div>
                        <div class="sidebar-section">
                            <div class="section-title">Today's Focus</div>
                            <div class="goals-panel">
                                ${scenario.goals.map(goal => `
                                    <div class="goal-item">${goal}</div>
                                `).join('')}
                            </div>
                        </div>
                    </div>

                    <!-- Center: Chat Interface -->
                    <div class="center-chat">
                        <div class="chat-messages-wrapper">
                            ${scenario.messages.map(msg => `
                                <div class="chat-message message-${msg.type}">
                                    <div class="message-label">${msg.type === 'user' ? 'You' : 'Luna9'}</div>
                                    <div class="message-bubble">${this.formatMessage(msg.text)}</div>
                                </div>
                            `).join('')}
                        </div>

                        <!-- Chat Input (visual only) -->
                        <div class="chat-input">
                            <div class="chat-input-field">Type your message...</div>
                            <button class="chat-send-btn">↑</button>
                        </div>
                    </div>

                    <!-- Right Sidebar: Project Graph -->
                    <div class="right-sidebar">
                        <div class="section-title">Project Context</div>
                        <div class="project-graph">
                            ${scenario.graph.map(node => `
                                <div class="graph-node ${node.active ? 'active' : ''}">
                                    <div class="node-label">${node.label}</div>
                                    ${node.active ? '<small>Active context</small>' : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderFolders(folders) {
        return folders.map(folder => `
            <div class="folder-item">${folder.name}</div>
            ${folder.files.map(file => `
                <div class="file-item">${file}</div>
            `).join('')}
        `).join('');
    }

    formatMessage(text) {
        // Convert newlines to <br> for display
        return text.replace(/\n/g, '<br>');
    }

    attachEventListeners() {
        // Scenario switcher
        this.container.querySelectorAll('.mockup-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const scenario = e.target.dataset.scenario;
                if (scenario && scenario !== this.currentScenario) {
                    this.currentScenario = scenario;
                    this.render();
                    this.attachEventListeners();
                }
            });
        });

        // Theme toggle
        const themeToggle = this.container.querySelector('#theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
                this.render();
                this.attachEventListeners();
            });
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const mockupContainer = document.getElementById('desktop-mockup-container');
    if (mockupContainer) {
        new DesktopMockup('desktop-mockup-container');
    }
});
