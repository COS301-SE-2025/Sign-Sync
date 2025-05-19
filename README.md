
# Apollo Projects - Sign Sync

**What It Does:**

Sign Sync is an innovative AI-powered communication bridge that enables seamless two-way translation between spoken language and sign language. Using speech recognition, motion capture, and AI animation, it converts:

- Speech to Sign Language: Translates spoken words into real-time animated sign language gestures.

- Sign to Speech/Text: Captures sign language movements and converts them into spoken words or text.

Built as a **web application**, it provides an intuitive, accessible platform for instant communication between deaf/hard-of-hearing individuals and hearing individuals.

**Who It’s For:**

- Deaf & Hard-of-Hearing Community – Enables effortless communication without needing an interpreter.

- Hearing Individuals – Helps those unfamiliar with sign language engage in inclusive conversations.

- Educational & Workplace Settings – Supports accessibility in schools, offices, and public services.

- Developers & Tech Enthusiasts – A cutting-edge project exploring AI, real-time animation, and human-computer interaction.

**Why It Matters:**

Sign Sync isn’t just a tool—it’s a barrier-breaking solution that promotes inclusivity, independence, and seamless communication. By blending advanced tech with real-world impact, this project empowers users while offering developers a chance to work on meaningful, next-gen AI systems.

## Links

- [Functional Requirements (SRS)]()
- [GitHub Project Board](https://github.com/COS301-SE-2025/Sign-Sync/projects)

## Team

|**Name**|**Discription**|**LinkedIn**|
|--------|---------------|----------|
| Michael Stone | | [LinkedIn]()
| Matthew Gravette | | [LinkedIn]()
| Wj van der Walt | | [LinkedIn]()
| Jamean Groenewald | | [LinkedIn]()
| Stefan Müller | | [LinkedIn]()

## Git Repository Structure

### Repository Type: Monorepo
This project follows a **monorepo structure**, where all components are maintained in a single repository. The structure is as follows:

### Git Branching Strategy
This repository follows a structured branching strategy:
- `main` - The stable production-ready branch
- `develop` - Integration branch for ongoing development work
- `feature/*` - Each new feature is developed in its own branch (e.g., `feature/HandTracking`)
- `bugfix/*` - Dedicated branches for bugfixes (e.g., `bugfix/Tracking-errors`)
- `release/*` - Used for versioned releases (e.g., `release/v1.0`)
- `UI/*` - Dedicated branches for UI devolopment (e.g., `UI/LoginPage`)

### Git Organization and Management
- Changes are introduced through **feature branches** and merged into `main` via pull requests
- Code reviews are performed before merging
- Releases follow semantic versioning (`release/v1.0`, `release/v1.1`, etc.)
- Branch naming follows a consistent pattern for easy identification (feature/, bugfix/, release/)
- Temporary test branches like `Tracking_Test` are used for isolated testing
- Branches are regularly updated to stay in sync with main, as shown by the "Behind/Ahead" metrics

### Code Quality Badges

| Category          | Badges |
|-------------------|--------|
| **Code Quality**  | ![Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen) ![SonarCloud](https://img.shields.io/badge/SonarCloud-Passed-success) |
| **Build**         | ![GitHub Actions](https://img.shields.io/github/actions/workflow/status/StefanDramaLlama/sign-sync/main.yml?label=Build) |
| **Dependencies**  | ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) |
| **Community**     | ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen) ![GitHub Issues](https://img.shields.io/github/issues/StefanDramaLlama/sign-sync) |
