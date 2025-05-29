
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

## Demos

||**Demo Slides**|**Demo Video**|
|-|-|-|
|Demo 1| [Slides 1](https://docs.google.com/presentation/d/1dGRHy4usAEJKe0lt4n1SQhi-hZZ6lwPh31kqmcY2CO8/edit?slide=id.g35dd9153325_0_0#slide=id.g35dd9153325_0_0) | [Video 1]()
|Demo 2| [Slides 2]() | [Video 2]()
|Demo 3| [Slides 3]() | [Video 3]()
|Demo 4| [Slides 4]() | [Video 4]()

## Documentation Links

|**Document**|**Description**|
|-|-|
|[SRS Requirements](https://docs.google.com/document/d/1pcefn-Nhll3uHIxUDYLyK4dad5k8ZgvDUs-wr97H06Q/edit?tab=t.0#heading=h.tbpn49glggld)| Software Requiremtn Specifications |
|[User Manual]()| Guide for users on how to use the application |
|[Technical Installation Manual]()| Instructions for setting up the applications |
|[Testing Policy]()| Policies and procedures for testing |
|[Coding Standards]()| Guidelines for coding pracices |
|[Advertisment Video]()| Advertisment for Sign Sync |

- [GitHub Project Board](https://github.com/COS301-SE-2025/Sign-Sync/projects)

## Team

|**Name**|**Discription**|**LinkedIn**|
|--------|---------------|----------|
| Michael Stone | Michael is an enthusiastic final-year BSc Information and Knowledge Systems student. With a strong foundation in both Computer Science and Informatics, he possesses a well-rounded skill set and a deep interest in full-stack development. His versatility enables him to contribute effectively to frontend, backend, and system integration projects. He has built several web applications and APIs, and has experience integrating them with backend systems. He enjoys working individually and alongside others, sharing ideas, and contributing to team success. Even when faced with unfamiliar challenges, Michael is committed to finding effective solutions through persistence, research, and a willingness to learn. | [LinkedIn](https://www.linkedin.com/in/michael-stone-7209b3216/)
| Matthew Gravette | Matthew is a final year BSc Computer Science student, he has completed many projects, big and small. He has extensive experience in both C++ and Java. Matthew has a deep interest in full stack web development, having built a website for auctions that could handle multiple concurrent bidders and a progressive web app for sound engineers. He strives to bring all these skills to help the team with the development of this project. He has worked in many teams of various sizes and has played both a managing role and a supporting role. He is experienced in all standard web development languages and , He is also proficient in data science and does 3rd year statistics and is well versed in database creation and management. | [LinkedIn](https://www.linkedin.com/in/matthew-gravette-0a32402ab/)
| Wj van der Walt | W.J. is a devoted full-time student with experience in full-stack development and machine learning. He is a dedicated and productive developer who leverages a wide range of tools, including GitHub, to enhance his workflow. W.J. thrives both independently and in collaborative environments, having successfully completed numerous group projects. While he enjoys all aspects of development, his true passion lies in leveraging machine learning, AI, and backend systems to solve real-world problems. As a fast and eager learner, he is confident in his ability to tackle any task assigned to him and his team. | [LinkedIn](https://www.linkedin.com/in/wj-van-der-walt-a1171a22b)
| Jamean Groenewald | Jamean is a final year BSc Information and Knowledge Systems student with a strong focus on frontend development. He has a passion for crafting intuitive and responsive user interfaces. His degree has provided a solid foundation in web development principles, and he enjoys working with modern frameworks and tools to bring ideas to life. Jamean has built several web interfaces that communicate seamlessly with RESTful APIs, emphasizing user experience and performance. He is an enthusiastic team player who thrives in collaborative settings, having worked on multiple university projects that required clear communication, shared problem-solving, and the integration of different technical components. For this project, Jamean is especially excited to contribute to the frontend architecture and API integration, ensuring the user-facing side of the application is both functional and engaging. | [LinkedIn](https://www.linkedin.com/in/jamean-groenewald-a334a5356/)
| Stefan Müller | Stefan is a versatile third-year Computer Science student with a growing interest in frontend development, system integration, and AI. While he has experience in backend technologies. His coursework has equipped him with proficiency in JavaScript, C++, and Java. Recently, he has been exploring AI and machine learning. Stefan thrives in collaborative environments, having contributed to a university group project where he worked on both frontend and backend integration for a No-SQL Database. He enjoys bridging the gap between different system components, ensuring smooth data flow and functionality. For this project, he is particularly excited to work on frontend development, API integration, and potentially AI-driven features, bringing a well-rounded perspective to the team. | [LinkedIn](https://www.linkedin.com/in/stefan-m%C3%BCller-29b3b2296/)

## Git Repository Structure

### Repository Type: Monorepo
This project follows a **monorepo structure**, where all components are maintained in a single repository. The structure is as follows:

### Git Branching Strategy
This repository follows a structured branching strategy:
- `main` - The stable production-ready branch
- `develop` - Integration branch for ongoing development work
- `develop/frontend/` - Integration branch for frontend-specific features.
- `develop/frontend/feature/*` - Short-lived branches for individual frontend features.
- `develop/backend/` - Integration branch for backend-specific features.
- `develop/backend/feature/*` - Short-lived branches for individual backend features.
- `release/*` - Temporary branches for final testing before production.
- `hotfix/*` - Emergency branches for critical production bug fixes.

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
