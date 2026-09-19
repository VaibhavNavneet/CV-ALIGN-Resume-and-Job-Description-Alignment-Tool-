"""Golden evaluation dataset for the evidence-retrieval (RAG) layer.

Five synthetic (resume, JD) pairs across different engineering domains
(backend, ML, frontend, mobile, DevOps), each resume containing one
genuinely relevant item and one deliberate decoy per section (experience,
projects) so Recall@K is non-trivial to compute (not just 1-of-1). Two
negative-control queries pair a resume against an unrelated JD, where the
correct behaviour is to retrieve nothing -- used to check the retriever
doesn't fabricate relevance where none exists.

Relevance is hand-labelled via a predicate over each chunk's *reconstructed*
text (what `pipeline/retrieval.py::build_chunks` produces), not the resume
model itself, so labels stay correct even if chunk-text formatting changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from multi_agent_resume_screener.state import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    Section,
    StructuredJD,
    StructuredResume,
)


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    resume: StructuredResume
    jd: StructuredJD
    section: Section
    # True for chunk text that should count as relevant evidence for this
    # query. A query where no chunk in `resume` satisfies this is a
    # negative control (expected retrieval: nothing).
    relevant: Callable[[str], bool]


# --------------------------------------------------------------------------- #
# Resumes (one relevant + one decoy item per section, by design)
# --------------------------------------------------------------------------- #
BACKEND_RESUME = StructuredResume(
    name="Priya Sharma",
    skills=["Python", "FastAPI", "PostgreSQL", "Git", "Docker", "REST APIs", "HTML", "CSS"],
    experience=[
        ExperienceItem(
            company="PayEase Fintech",
            role="Backend Developer Intern",
            dates="2024",
            bullets=[
                "Built REST APIs in Python using FastAPI for a payments platform",
                "Designed PostgreSQL schemas and optimized slow queries",
            ],
        ),
        ExperienceItem(
            company="TechFest College Club",
            role="Campus Ambassador",
            dates="2023",
            bullets=["Promoted college tech fest on social media", "Coordinated event logistics"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Expense Tracker API",
            description="A FastAPI backend service with PostgreSQL storage for tracking personal expenses",
            tech=["FastAPI", "PostgreSQL", "Docker"],
        ),
        ProjectItem(
            name="College Cultural Fest Website",
            description="A static HTML/CSS landing page for the college's annual cultural fest",
            tech=["HTML", "CSS"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="NIT Trichy", year="2024")],
)

ML_RESUME = StructuredResume(
    name="Rahul Verma",
    skills=["Python", "PyTorch", "scikit-learn", "Pandas", "NumPy", "SQL", "Git"],
    experience=[
        ExperienceItem(
            company="VisionAI Labs",
            role="Machine Learning Intern",
            dates="2024",
            bullets=[
                "Trained CNN models in PyTorch for image classification",
                "Built data preprocessing pipelines with Pandas and NumPy",
            ],
        ),
        ExperienceItem(
            company="Retail Mart",
            role="Sales Associate",
            dates="2022-2023",
            bullets=["Assisted customers on the sales floor", "Managed inventory stock counts"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Sentiment Analysis on Twitter Data",
            description="An NLP pipeline using scikit-learn and TF-IDF to classify tweet sentiment",
            tech=["scikit-learn", "Python", "NLP"],
        ),
        ProjectItem(
            name="Personal Portfolio Website",
            description="A static personal portfolio site built with HTML, CSS, and vanilla JS",
            tech=["HTML", "CSS", "JavaScript"],
        ),
    ],
    education=[EducationItem(degree="B.Sc Statistics", institute="Delhi University", year="2023")],
)

FRONTEND_RESUME = StructuredResume(
    name="Ananya Iyer",
    skills=["JavaScript", "React", "Redux", "HTML", "CSS", "Tailwind CSS", "Git"],
    experience=[
        ExperienceItem(
            company="ShopEasy",
            role="Frontend Developer Intern",
            dates="2024",
            bullets=[
                "Built responsive UI components in React",
                "Managed application state with Redux and integrated REST APIs",
            ],
        ),
        ExperienceItem(
            company="Local NGO",
            role="Volunteer Teacher",
            dates="2023",
            bullets=["Taught basic computer literacy to schoolchildren"],
        ),
    ],
    projects=[
        ProjectItem(
            name="E-commerce Storefront UI",
            description="A React and Tailwind CSS storefront connected to a mock REST API",
            tech=["React", "Tailwind CSS", "Redux"],
        ),
        ProjectItem(
            name="Weather CLI Tool",
            description="A Python command-line script that fetches and prints weather forecasts",
            tech=["Python"],
        ),
    ],
    education=[EducationItem(degree="B.E. Information Technology", institute="Anna University", year="2024")],
)

MOBILE_RESUME = StructuredResume(
    name="Karan Mehta",
    skills=["Kotlin", "Android SDK", "Jetpack Compose", "Retrofit", "Room", "Git"],
    experience=[
        ExperienceItem(
            company="AppForge Studios",
            role="Android Developer Intern",
            dates="2024",
            bullets=[
                "Built Android app features in Kotlin using Jetpack Compose",
                "Integrated Retrofit for REST networking and Room for local storage",
            ],
        ),
        ExperienceItem(
            company="Freelance",
            role="Content Writer",
            dates="2022-2023",
            bullets=["Wrote blog articles and optimized them for SEO"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Habit Tracker Android App",
            description="A Kotlin Android app using Jetpack Compose and Room for local persistence",
            tech=["Kotlin", "Jetpack Compose", "Room"],
        ),
        ProjectItem(
            name="Excel Automation Macro",
            description="A VBA macro that automates monthly expense report formatting in Excel",
            tech=["VBA"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Electronics and Communication", institute="VIT Vellore", year="2023")],
)

DEVOPS_RESUME = StructuredResume(
    name="Sneha Nair",
    skills=["Docker", "Kubernetes", "AWS", "GitHub Actions", "Terraform", "Linux", "Git"],
    experience=[
        ExperienceItem(
            company="CloudNine Systems",
            role="DevOps Intern",
            dates="2024",
            bullets=[
                "Set up CI/CD pipelines with GitHub Actions and Docker",
                "Deployed and managed containerized services on AWS EC2",
            ],
        ),
        ExperienceItem(
            company="City Library",
            role="Front Desk Assistant",
            dates="2022",
            bullets=["Assisted visitors and managed book checkouts"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Kubernetes Cluster Monitoring Dashboard",
            description="A self-hosted Kubernetes cluster monitored with Prometheus and Grafana",
            tech=["Kubernetes", "Prometheus", "Grafana"],
        ),
        ProjectItem(
            name="Recipe Sharing Mobile App",
            description="A Flutter mobile app for sharing and browsing recipes",
            tech=["Flutter", "Dart"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="IIIT Hyderabad", year="2022")],
)

DATA_ENGINEERING_RESUME = StructuredResume(
    name="Arjun Malhotra",
    skills=["Python", "SQL", "Apache Airflow", "Apache Spark", "dbt", "AWS", "Git", "Excel"],
    experience=[
        ExperienceItem(
            company="StreamWorks Analytics",
            role="Data Engineer Intern",
            dates="2024",
            bullets=[
                "Built and orchestrated ETL pipelines with Apache Airflow processing 10M+ records daily with automated data-quality checks",
                "Optimized Spark jobs on AWS EMR for large-scale data transformation, cutting batch processing time by 45%",
            ],
        ),
        ExperienceItem(
            company="Bright Minds Tutoring Center",
            role="Math Tutor",
            dates="2022-2023",
            bullets=["Tutored high school students in algebra and calculus", "Prepared practice worksheets for weekly sessions"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Real-Time Data Pipeline with Kafka and Spark",
            description="A streaming ETL pipeline using Apache Kafka and Spark Structured Streaming to ingest and transform clickstream data",
            tech=["Apache Kafka", "Apache Spark", "Python"],
        ),
        ProjectItem(
            name="Recipe Blog Static Site",
            description="A static blog built with Jekyll and hosted on GitHub Pages",
            tech=["Jekyll", "HTML"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="BITS Pilani", year="2024")],
)

QA_RESUME = StructuredResume(
    name="Divya Krishnan",
    skills=["Selenium", "Python", "TestNG", "Java", "Postman", "JIRA", "Git", "PowerPoint"],
    experience=[
        ExperienceItem(
            company="BugFree Software",
            role="QA Automation Engineer Intern",
            dates="2024",
            bullets=[
                "Designed and maintained a Selenium WebDriver suite of 150+ automated UI test cases in Java, improving test coverage by 40%",
                "Built API test scripts using Postman and integrated them into the CI/CD pipeline, catching regressions before production",
            ],
        ),
        ExperienceItem(
            company="City Marathon Committee",
            role="Event Volunteer",
            dates="2023",
            bullets=["Coordinated volunteer schedules for a citywide marathon event", "Managed registration desk on event day"],
        ),
    ],
    projects=[
        ProjectItem(
            name="E-commerce Regression Test Suite",
            description="An automated regression suite using Selenium and TestNG covering checkout and payment flows",
            tech=["Selenium", "TestNG", "Java"],
        ),
        ProjectItem(
            name="Budget Planner Spreadsheet",
            description="A personal budgeting spreadsheet template built in Excel with macros",
            tech=["Excel", "VBA"],
        ),
    ],
    education=[EducationItem(degree="B.E. Computer Science", institute="PES University", year="2024")],
)

CYBERSECURITY_RESUME = StructuredResume(
    name="Vikram Chatterjee",
    skills=["Python", "Burp Suite", "OWASP", "Nmap", "Linux", "Cryptography", "Git", "Photoshop"],
    experience=[
        ExperienceItem(
            company="SecureNet Labs",
            role="Application Security Intern",
            dates="2024",
            bullets=[
                "Performed penetration testing and vulnerability assessments on internal web applications using Burp Suite and Nmap",
                "Identified and helped remediate OWASP Top 10 vulnerabilities in production web applications, reducing high-severity findings by 40%",
            ],
        ),
        ExperienceItem(
            company="FreshMart Grocery",
            role="Retail Associate",
            dates="2022-2023",
            bullets=["Handled billing and customer checkout", "Restocked shelves and managed inventory"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Web App Vulnerability Scanner",
            description="A Python tool that automates OWASP Top 10 vulnerability scanning for web applications",
            tech=["Python", "OWASP", "Burp Suite"],
        ),
        ProjectItem(
            name="Photo Editing Portfolio Site",
            description="A personal portfolio site showcasing photo editing work built with WordPress",
            tech=["WordPress", "Photoshop"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Information Security", institute="SRM Institute of Science and Technology", year="2024")],
)

SRE_RESUME = StructuredResume(
    name="Rohan Deshpande",
    skills=["Linux", "Kubernetes", "Prometheus", "Go", "Bash", "AWS", "Git", "MS Word"],
    experience=[
        ExperienceItem(
            company="Uptime Cloud Co.",
            role="Site Reliability Engineer Intern",
            dates="2024",
            bullets=[
                "On-call for production incident response, maintaining 99.9% uptime SLA",
                "Built a Prometheus and Grafana observability stack that cut mean time to detection from 20 minutes to under 5 minutes",
            ],
        ),
        ExperienceItem(
            company="QuickBite Delivery",
            role="Delivery Driver",
            dates="2022",
            bullets=["Delivered food orders across the city on a two-wheeler", "Maintained delivery time logs"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Incident Response Automation Bot",
            description="A Go-based Slack bot that automates on-call incident response and paging using Prometheus alerts",
            tech=["Go", "Prometheus", "Kubernetes"],
        ),
        ProjectItem(
            name="Personal Blog CMS",
            description="A simple content management system for a personal blog built with PHP and MySQL",
            tech=["PHP", "MySQL"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="IIT Guwahati", year="2023")],
)

FULLSTACK_RESUME = StructuredResume(
    name="Neha Bansal",
    skills=["JavaScript", "Node.js", "Express", "MongoDB", "React", "Git", "REST APIs", "Illustrator"],
    experience=[
        ExperienceItem(
            company="CartHub Retailtech",
            role="Full-Stack Developer Intern",
            dates="2024",
            bullets=[
                "Built REST APIs with Node.js and Express backed by MongoDB",
                "Developed React front-end components consuming those APIs",
            ],
        ),
        ExperienceItem(
            company="WriteRight Media",
            role="Content Writer",
            dates="2023",
            bullets=["Wrote SEO-optimized blog articles for client websites", "Edited copy for marketing newsletters"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Task Management Web App",
            description="A full-stack task manager with a Node.js/Express backend, MongoDB storage, and a React frontend",
            tech=["Node.js", "Express", "MongoDB", "React"],
        ),
        ProjectItem(
            name="Mobile Game Prototype",
            description="A 2D platformer game prototype built in Unity with C#",
            tech=["Unity", "C#"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Information Technology", institute="Manipal Institute of Technology", year="2024")],
)

EMBEDDED_RESUME = StructuredResume(
    name="Aditya Rao",
    skills=["C", "C++", "Embedded C", "ARM Cortex", "RTOS", "I2C/SPI", "Git", "PowerPoint"],
    experience=[
        ExperienceItem(
            company="ChipWorks Electronics",
            role="Embedded Systems Intern",
            dates="2024",
            bullets=[
                "Developed firmware in Embedded C for ARM Cortex-M microcontrollers",
                "Implemented I2C and SPI communication protocols for sensor interfacing",
            ],
        ),
        ExperienceItem(
            company="Grand Palace Hotel",
            role="Front Desk Assistant",
            dates="2022",
            bullets=["Greeted guests and managed check-in/check-out", "Handled phone reservations"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Smart Home IoT Gateway",
            description="An IoT gateway running FreeRTOS on an ARM Cortex-M microcontroller, communicating with sensors over I2C and SPI",
            tech=["FreeRTOS", "ARM Cortex-M", "C"],
        ),
        ProjectItem(
            name="Instagram Analytics Dashboard",
            description="A web dashboard that visualizes Instagram engagement metrics using Chart.js",
            tech=["JavaScript", "Chart.js"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Electronics and Communication", institute="IIT Roorkee", year="2024")],
)

GAMEDEV_RESUME = StructuredResume(
    name="Ishaan Kapoor",
    skills=["C#", "Unity", "Unity Engine", "Blender", "Git", "Shader Programming", "Photoshop", "Excel"],
    experience=[
        ExperienceItem(
            company="PixelForge Games",
            role="Game Developer Intern",
            dates="2024",
            bullets=[
                "Developed gameplay mechanics and physics systems in Unity using C#",
                "Optimized shader performance for mobile game builds",
            ],
        ),
        ExperienceItem(
            company="Central Public Library",
            role="Library Assistant",
            dates="2022-2023",
            bullets=["Catalogued and shelved returned books", "Assisted patrons with library resources"],
        ),
    ],
    projects=[
        ProjectItem(
            name="2D Platformer Adventure Game",
            description="A 2D platformer built in Unity with C# scripting, custom physics, and level design tools",
            tech=["Unity", "C#"],
        ),
        ProjectItem(
            name="Personal Finance Tracker",
            description="A Python command-line app for tracking monthly expenses and savings goals",
            tech=["Python"],
        ),
    ],
    education=[EducationItem(degree="B.Sc Game Design and Development", institute="DSK Supinfogame", year="2023")],
)

BLOCKCHAIN_RESUME = StructuredResume(
    name="Siddharth Oberoi",
    skills=["Solidity", "Ethereum", "Web3.js", "Hardhat", "JavaScript", "Git", "Smart Contracts", "Excel"],
    experience=[
        ExperienceItem(
            company="ChainForge Labs",
            role="Blockchain Developer Intern",
            dates="2024",
            bullets=[
                "Wrote and deployed Solidity smart contracts on the Ethereum testnet, reducing gas costs by 20% across core contracts",
                "Built a Web3.js frontend for interacting with deployed contracts",
            ],
        ),
        ExperienceItem(
            company="Homework Help Center",
            role="Tutor",
            dates="2022",
            bullets=["Tutored middle school students in mathematics", "Graded weekly assignments"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Decentralized Voting DApp",
            description="A decentralized voting application with Solidity smart contracts deployed via Hardhat and a Web3.js frontend",
            tech=["Solidity", "Hardhat", "Web3.js"],
        ),
        ProjectItem(
            name="Restaurant Menu Website",
            description="A static restaurant menu website built with HTML and CSS",
            tech=["HTML", "CSS"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="Thapar Institute of Engineering and Technology", year="2024")],
)

CLOUD_ARCHITECT_RESUME = StructuredResume(
    name="Meera Subramaniam",
    skills=["AWS", "Terraform", "CloudFormation", "AWS Lambda", "VPC Design", "Python", "Git", "Canva"],
    experience=[
        ExperienceItem(
            company="SkyStack Cloud Consulting",
            role="Cloud Solutions Architect Intern",
            dates="2024",
            bullets=[
                "Designed multi-tier AWS architectures using VPCs, Lambda, and CloudFormation for client migrations",
                "Authored reusable Terraform modules to provision infrastructure as code, cutting environment setup time by 35%",
            ],
        ),
        ExperienceItem(
            company="City Heritage Museum",
            role="Museum Guide",
            dates="2022",
            bullets=["Led guided tours for visiting school groups", "Answered visitor questions about exhibits"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Multi-Region AWS Landing Zone",
            description="A reusable Terraform-based AWS landing zone with VPC, IAM, and CloudFormation stacks for multi-account setups",
            tech=["Terraform", "AWS", "CloudFormation"],
        ),
        ProjectItem(
            name="Podcast Discovery App",
            description="A Flutter mobile app for discovering and subscribing to podcasts",
            tech=["Flutter", "Dart"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="IIT Bombay", year="2023")],
)

IOS_RESUME = StructuredResume(
    name="Tanvi Joshi",
    skills=["Swift", "SwiftUI", "Xcode", "iOS SDK", "Core Data", "Git", "REST APIs", "Excel"],
    experience=[
        ExperienceItem(
            company="AppOrbit Studios",
            role="iOS Developer Intern",
            dates="2024",
            bullets=[
                "Built iOS app screens in SwiftUI consuming REST APIs",
                "Used Core Data for local persistence and Xcode Instruments for performance profiling",
            ],
        ),
        ExperienceItem(
            company="Brew & Bean Cafe",
            role="Barista",
            dates="2022-2023",
            bullets=["Prepared and served coffee orders", "Managed cash register and daily inventory"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Fitness Tracking iOS App",
            description="A SwiftUI iOS app for tracking workouts, using Core Data for local storage",
            tech=["Swift", "SwiftUI", "Core Data"],
        ),
        ProjectItem(
            name="Inventory Management Desktop App",
            description="A Java Swing desktop application for small-business inventory management",
            tech=["Java", "Swing"],
        ),
    ],
    education=[EducationItem(degree="B.E. Computer Engineering", institute="Pune Institute of Computer Technology", year="2024")],
)

DBA_RESUME = StructuredResume(
    name="Kabir Singh",
    skills=["PostgreSQL", "MySQL", "Oracle DB", "SQL", "Database Tuning", "Linux", "Git", "PowerPoint"],
    experience=[
        ExperienceItem(
            company="DataVault Systems",
            role="Database Administrator Intern",
            dates="2024",
            bullets=[
                "Performed query optimization and index tuning across PostgreSQL and MySQL databases, cutting slow-query latency by 30%",
                "Managed database backups, replication, and disaster recovery procedures",
            ],
        ),
        ExperienceItem(
            company="SwiftShip Logistics",
            role="Warehouse Associate",
            dates="2022",
            bullets=["Picked and packed customer orders", "Operated forklift for pallet movement"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Database Performance Monitoring Tool",
            description="A tool that tracks PostgreSQL query performance and automates index tuning recommendations",
            tech=["PostgreSQL", "SQL", "Python"],
        ),
        ProjectItem(
            name="Wedding Invitation Website",
            description="A static wedding invitation website built with HTML and CSS",
            tech=["HTML", "CSS"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="Jadavpur University", year="2023")],
)

DEVREL_RESUME = StructuredResume(
    name="Ritika Bhatt",
    skills=["Python", "REST APIs", "Technical Writing", "JIRA", "Zendesk", "SQL", "Git", "Photoshop"],
    experience=[
        ExperienceItem(
            company="APIHub Inc.",
            role="Developer Relations Engineer Intern",
            dates="2024",
            bullets=[
                "Wrote developer documentation and API reference guides for a public REST API, surfacing and closing documentation gaps",
                "Resolved technical support tickets and reproduced customer-reported API bugs, cutting first-response time by 30%",
            ],
        ),
        ExperienceItem(
            company="FitZone Gym",
            role="Gym Receptionist",
            dates="2022",
            bullets=["Greeted members and managed class bookings", "Handled membership renewals"],
        ),
    ],
    projects=[
        ProjectItem(
            name="API Documentation Portal",
            description="A developer documentation portal with interactive REST API examples and code snippets",
            tech=["Python", "REST APIs", "Markdown"],
        ),
        ProjectItem(
            name="Wildlife Photography Blog",
            description="A personal blog showcasing wildlife photography built with WordPress",
            tech=["WordPress"],
        ),
    ],
    education=[EducationItem(degree="B.Sc Computer Science", institute="Christ University", year="2024")],
)

COMPUTER_VISION_RESUME = StructuredResume(
    name="Aryan Kulkarni",
    skills=["Python", "OpenCV", "PyTorch", "YOLO", "Image Processing", "NumPy", "Git", "Excel"],
    experience=[
        ExperienceItem(
            company="OptiSight Robotics",
            role="Computer Vision Intern",
            dates="2024",
            bullets=[
                "Built real-time object detection pipelines using YOLO and OpenCV, improving detection accuracy by 15% for camera-based defect detection",
                "Implemented image preprocessing and augmentation pipelines that reduced inference time by 20%",
            ],
        ),
        ExperienceItem(
            company="ConnectSupport BPO",
            role="Call Center Agent",
            dates="2022-2023",
            bullets=["Handled inbound customer service calls", "Logged call resolutions in a CRM system"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Real-Time Face Mask Detection System",
            description="A real-time face mask detection system using OpenCV and a YOLO object detection model",
            tech=["OpenCV", "YOLO", "Python"],
        ),
        ProjectItem(
            name="Online Bookstore Inventory System",
            description="A relational database-backed inventory system for an online bookstore built with PHP",
            tech=["PHP", "MySQL"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="IIT Kharagpur", year="2024")],
)

GO_RUST_RESUME = StructuredResume(
    name="Devansh Trivedi",
    skills=["Go", "Rust", "gRPC", "Distributed Systems", "Linux", "Git", "Docker", "PowerPoint"],
    experience=[
        ExperienceItem(
            company="ScaleForge Systems",
            role="Backend Systems Engineer Intern",
            dates="2024",
            bullets=[
                "Built Go microservices using gRPC for inter-service communication, handling 5,000+ requests per second at peak",
                "Optimized a Rust-based data processing service, reducing p95 latency from 180ms to 95ms",
            ],
        ),
        ExperienceItem(
            company="TechConf Meetup Group",
            role="Event Volunteer",
            dates="2023",
            bullets=["Set up registration booths for a local tech conference", "Coordinated speaker schedules"],
        ),
    ],
    projects=[
        ProjectItem(
            name="Distributed Key-Value Store",
            description="A distributed key-value store written in Go, using gRPC for replication and a Raft consensus implementation",
            tech=["Go", "gRPC", "Distributed Systems"],
        ),
        ProjectItem(
            name="Recipe Recommendation Website",
            description="A Django-based website that recommends recipes based on user preferences",
            tech=["Django", "Python"],
        ),
    ],
    education=[EducationItem(degree="B.Tech Computer Science", institute="IIT Delhi", year="2023")],
)


# --------------------------------------------------------------------------- #
# Job descriptions
# --------------------------------------------------------------------------- #
JD_BACKEND = StructuredJD(
    title="Backend Engineer",
    required_skills=["Python", "FastAPI", "PostgreSQL", "REST APIs"],
    nice_to_have_skills=["Docker"],
    responsibilities=["Design and build REST APIs", "Work with PostgreSQL databases"],
    qualifications=["Bachelor's degree in Computer Science or a related field"],
)
JD_ML = StructuredJD(
    title="Machine Learning Engineer",
    required_skills=["Python", "PyTorch", "scikit-learn", "Pandas"],
    nice_to_have_skills=["NLP"],
    responsibilities=["Train and evaluate machine learning models", "Build data preprocessing pipelines"],
    qualifications=["Degree in Statistics, Mathematics, Computer Science, or a related field"],
)
JD_FRONTEND = StructuredJD(
    title="Frontend Engineer",
    required_skills=["JavaScript", "React", "HTML", "CSS"],
    nice_to_have_skills=["Redux", "Tailwind CSS"],
    responsibilities=["Build responsive user interfaces in React", "Integrate the frontend with REST APIs"],
)
JD_MOBILE = StructuredJD(
    title="Android Engineer",
    required_skills=["Kotlin", "Android SDK", "Jetpack Compose"],
    nice_to_have_skills=["Retrofit", "Room"],
    responsibilities=["Develop native Android features in Kotlin", "Work with local databases and REST clients"],
)
JD_DEVOPS = StructuredJD(
    title="DevOps Engineer",
    required_skills=["Docker", "Kubernetes", "AWS"],
    nice_to_have_skills=["Terraform", "GitHub Actions"],
    responsibilities=["Build and maintain CI/CD pipelines", "Manage containerized deployments on AWS"],
)
JD_DATA_ENGINEERING = StructuredJD(
    title="Data Engineer",
    required_skills=["Python", "SQL", "Apache Airflow", "Apache Spark"],
    nice_to_have_skills=["dbt", "AWS"],
    responsibilities=["Build and maintain ETL pipelines", "Optimize large-scale data processing jobs"],
    qualifications=["Bachelor's degree in Computer Science, Data Engineering, or a related field"],
)
JD_QA = StructuredJD(
    title="QA Automation Engineer",
    required_skills=["Selenium", "Java", "TestNG"],
    nice_to_have_skills=["Postman", "JIRA"],
    responsibilities=["Design and maintain automated test suites", "Integrate tests into CI/CD pipelines"],
)
JD_CYBERSECURITY = StructuredJD(
    title="Application Security Engineer",
    required_skills=["Python", "OWASP", "Burp Suite"],
    nice_to_have_skills=["Nmap", "Cryptography"],
    responsibilities=["Conduct penetration tests and vulnerability assessments", "Remediate security vulnerabilities in web applications"],
    qualifications=["Bachelor's degree in Computer Science, Information Security, or a related field"],
)
JD_SRE = StructuredJD(
    title="Site Reliability Engineer",
    required_skills=["Linux", "Kubernetes", "Prometheus"],
    nice_to_have_skills=["Go", "AWS"],
    responsibilities=["Maintain production uptime and lead incident response", "Build monitoring and alerting systems"],
)
JD_FULLSTACK = StructuredJD(
    title="Full-Stack Engineer (Node.js)",
    required_skills=["Node.js", "Express", "MongoDB", "React"],
    nice_to_have_skills=["REST APIs"],
    responsibilities=["Build REST APIs with Node.js and Express", "Develop React-based front-ends"],
    qualifications=["Bachelor's degree in Computer Science, IT, or a related field"],
)
JD_EMBEDDED = StructuredJD(
    title="Embedded Systems Engineer",
    required_skills=["Embedded C", "ARM Cortex", "RTOS"],
    nice_to_have_skills=["I2C", "SPI"],
    responsibilities=["Develop firmware for microcontroller-based systems", "Interface sensors over I2C/SPI"],
)
JD_GAMEDEV = StructuredJD(
    title="Game Developer (Unity)",
    required_skills=["Unity", "C#"],
    nice_to_have_skills=["Shader Programming", "Blender"],
    responsibilities=["Implement gameplay mechanics in Unity", "Optimize game performance across platforms"],
    qualifications=["Degree in Game Design, Computer Science, or a related field"],
)
JD_BLOCKCHAIN = StructuredJD(
    title="Blockchain Developer",
    required_skills=["Solidity", "Ethereum", "Web3.js"],
    nice_to_have_skills=["Hardhat"],
    responsibilities=["Develop and deploy smart contracts", "Build frontends that interact with blockchain contracts"],
)
JD_CLOUD_ARCHITECT = StructuredJD(
    title="Cloud Solutions Architect",
    required_skills=["AWS", "Terraform", "CloudFormation"],
    nice_to_have_skills=["AWS Lambda"],
    responsibilities=["Design multi-tier cloud architectures", "Author infrastructure-as-code modules for cloud provisioning"],
    qualifications=["Bachelor's degree in Computer Science or a related field; AWS certification preferred"],
)
JD_IOS = StructuredJD(
    title="iOS Engineer",
    required_skills=["Swift", "SwiftUI", "iOS SDK"],
    nice_to_have_skills=["Core Data"],
    responsibilities=["Build native iOS app features in SwiftUI", "Work with local persistence and REST APIs"],
)
JD_DBA = StructuredJD(
    title="Database Administrator",
    required_skills=["PostgreSQL", "MySQL", "SQL"],
    nice_to_have_skills=["Database Tuning"],
    responsibilities=["Optimize database queries and indexes", "Manage backups and disaster recovery"],
    qualifications=["Bachelor's degree in Computer Science or a related field"],
)
JD_DEVREL = StructuredJD(
    title="Technical Support / Developer Relations Engineer",
    required_skills=["REST APIs", "Technical Writing", "Python"],
    nice_to_have_skills=["JIRA", "Zendesk"],
    responsibilities=["Write developer documentation and API guides", "Resolve technical support tickets from developers"],
)
JD_COMPUTER_VISION = StructuredJD(
    title="Computer Vision Engineer",
    required_skills=["Python", "OpenCV", "YOLO"],
    nice_to_have_skills=["Image Processing"],
    responsibilities=["Build object detection and image processing pipelines", "Deploy real-time computer vision models"],
    qualifications=["Bachelor's degree in Computer Science, Electronics, or a related field"],
)
JD_GO_RUST_SYSTEMS = StructuredJD(
    title="Backend Systems Engineer (Go/Rust)",
    required_skills=["Go", "Rust", "gRPC"],
    nice_to_have_skills=["Distributed Systems"],
    responsibilities=["Build high-throughput microservices in Go and Rust", "Design distributed systems using gRPC"],
)


# --------------------------------------------------------------------------- #
# Golden queries: (resume, matched JD, section) -> relevance predicate
# --------------------------------------------------------------------------- #
def _contains_any(*keywords: str) -> Callable[[str], bool]:
    return lambda text: any(kw.lower() in text.lower() for kw in keywords)


GOLDEN_QUERIES: list[GoldenQuery] = [
    # --- Backend / Priya ---
    GoldenQuery("backend_skills", BACKEND_RESUME, JD_BACKEND, "skills", _contains_any("FastAPI", "PostgreSQL")),
    GoldenQuery("backend_experience", BACKEND_RESUME, JD_BACKEND, "experience", _contains_any("FastAPI", "PostgreSQL")),
    GoldenQuery("backend_projects", BACKEND_RESUME, JD_BACKEND, "projects", _contains_any("FastAPI", "PostgreSQL")),
    GoldenQuery("backend_education", BACKEND_RESUME, JD_BACKEND, "education", _contains_any("Computer Science")),
    # --- ML / Rahul ---
    GoldenQuery("ml_skills", ML_RESUME, JD_ML, "skills", _contains_any("PyTorch", "scikit-learn", "Pandas")),
    GoldenQuery("ml_experience", ML_RESUME, JD_ML, "experience", _contains_any("PyTorch", "CNN", "Pandas")),
    GoldenQuery("ml_projects", ML_RESUME, JD_ML, "projects", _contains_any("scikit-learn", "NLP", "TF-IDF")),
    # --- Frontend / Ananya ---
    GoldenQuery("frontend_skills", FRONTEND_RESUME, JD_FRONTEND, "skills", _contains_any("React", "Redux")),
    GoldenQuery("frontend_experience", FRONTEND_RESUME, JD_FRONTEND, "experience", _contains_any("React", "Redux")),
    GoldenQuery("frontend_projects", FRONTEND_RESUME, JD_FRONTEND, "projects", _contains_any("React", "Tailwind")),
    # --- Mobile / Karan ---
    GoldenQuery("mobile_skills", MOBILE_RESUME, JD_MOBILE, "skills", _contains_any("Kotlin", "Jetpack Compose")),
    GoldenQuery("mobile_experience", MOBILE_RESUME, JD_MOBILE, "experience", _contains_any("Kotlin", "Jetpack Compose", "Retrofit")),
    GoldenQuery("mobile_projects", MOBILE_RESUME, JD_MOBILE, "projects", _contains_any("Kotlin", "Jetpack Compose", "Room")),
    # --- DevOps / Sneha ---
    GoldenQuery("devops_skills", DEVOPS_RESUME, JD_DEVOPS, "skills", _contains_any("Docker", "Kubernetes", "AWS")),
    GoldenQuery("devops_experience", DEVOPS_RESUME, JD_DEVOPS, "experience", _contains_any("CI/CD", "GitHub Actions", "AWS EC2")),
    GoldenQuery("devops_projects", DEVOPS_RESUME, JD_DEVOPS, "projects", _contains_any("Kubernetes", "Prometheus", "Grafana")),
    # --- Data Engineering / Arjun ---
    GoldenQuery("data_engineering_skills", DATA_ENGINEERING_RESUME, JD_DATA_ENGINEERING, "skills", _contains_any("Apache Airflow", "Apache Spark")),
    GoldenQuery("data_engineering_experience", DATA_ENGINEERING_RESUME, JD_DATA_ENGINEERING, "experience", _contains_any("Airflow", "Spark")),
    GoldenQuery("data_engineering_projects", DATA_ENGINEERING_RESUME, JD_DATA_ENGINEERING, "projects", _contains_any("Kafka", "Spark")),
    GoldenQuery("data_engineering_education", DATA_ENGINEERING_RESUME, JD_DATA_ENGINEERING, "education", _contains_any("Computer Science")),
    # --- QA Automation / Divya ---
    GoldenQuery("qa_skills", QA_RESUME, JD_QA, "skills", _contains_any("Selenium", "TestNG")),
    GoldenQuery("qa_experience", QA_RESUME, JD_QA, "experience", _contains_any("Selenium", "Postman")),
    GoldenQuery("qa_projects", QA_RESUME, JD_QA, "projects", _contains_any("Selenium", "TestNG")),
    # --- Cybersecurity / Vikram ---
    GoldenQuery("cybersecurity_skills", CYBERSECURITY_RESUME, JD_CYBERSECURITY, "skills", _contains_any("Burp Suite", "OWASP")),
    GoldenQuery("cybersecurity_experience", CYBERSECURITY_RESUME, JD_CYBERSECURITY, "experience", _contains_any("Burp Suite", "OWASP")),
    GoldenQuery("cybersecurity_projects", CYBERSECURITY_RESUME, JD_CYBERSECURITY, "projects", _contains_any("OWASP", "Burp Suite")),
    GoldenQuery("cybersecurity_education", CYBERSECURITY_RESUME, JD_CYBERSECURITY, "education", _contains_any("Information Security")),
    # --- SRE / Rohan ---
    GoldenQuery("sre_skills", SRE_RESUME, JD_SRE, "skills", _contains_any("Kubernetes", "Prometheus")),
    GoldenQuery("sre_experience", SRE_RESUME, JD_SRE, "experience", _contains_any("Prometheus", "uptime")),
    GoldenQuery("sre_projects", SRE_RESUME, JD_SRE, "projects", _contains_any("Prometheus", "Kubernetes")),
    # --- Full-Stack Node / Neha ---
    GoldenQuery("fullstack_skills", FULLSTACK_RESUME, JD_FULLSTACK, "skills", _contains_any("Node.js", "Express")),
    GoldenQuery("fullstack_experience", FULLSTACK_RESUME, JD_FULLSTACK, "experience", _contains_any("Node.js", "Express")),
    GoldenQuery("fullstack_projects", FULLSTACK_RESUME, JD_FULLSTACK, "projects", _contains_any("Node.js", "Express")),
    GoldenQuery("fullstack_education", FULLSTACK_RESUME, JD_FULLSTACK, "education", _contains_any("Information Technology")),
    # --- Embedded Systems / Aditya ---
    GoldenQuery("embedded_skills", EMBEDDED_RESUME, JD_EMBEDDED, "skills", _contains_any("Embedded C", "ARM Cortex")),
    GoldenQuery("embedded_experience", EMBEDDED_RESUME, JD_EMBEDDED, "experience", _contains_any("Embedded C", "ARM Cortex")),
    GoldenQuery("embedded_projects", EMBEDDED_RESUME, JD_EMBEDDED, "projects", _contains_any("ARM Cortex", "FreeRTOS")),
    # --- Game Development / Ishaan ---
    GoldenQuery("gamedev_skills", GAMEDEV_RESUME, JD_GAMEDEV, "skills", _contains_any("Unity", "C#")),
    GoldenQuery("gamedev_experience", GAMEDEV_RESUME, JD_GAMEDEV, "experience", _contains_any("Unity", "C#")),
    GoldenQuery("gamedev_projects", GAMEDEV_RESUME, JD_GAMEDEV, "projects", _contains_any("Unity", "C#")),
    GoldenQuery("gamedev_education", GAMEDEV_RESUME, JD_GAMEDEV, "education", _contains_any("Game Design")),
    # --- Blockchain / Siddharth ---
    GoldenQuery("blockchain_skills", BLOCKCHAIN_RESUME, JD_BLOCKCHAIN, "skills", _contains_any("Solidity", "Web3.js")),
    GoldenQuery("blockchain_experience", BLOCKCHAIN_RESUME, JD_BLOCKCHAIN, "experience", _contains_any("Solidity", "Ethereum")),
    GoldenQuery("blockchain_projects", BLOCKCHAIN_RESUME, JD_BLOCKCHAIN, "projects", _contains_any("Solidity", "Hardhat")),
    # --- Cloud Solutions Architect / Meera ---
    GoldenQuery("cloud_architect_skills", CLOUD_ARCHITECT_RESUME, JD_CLOUD_ARCHITECT, "skills", _contains_any("Terraform", "CloudFormation")),
    GoldenQuery("cloud_architect_experience", CLOUD_ARCHITECT_RESUME, JD_CLOUD_ARCHITECT, "experience", _contains_any("Terraform", "CloudFormation")),
    GoldenQuery("cloud_architect_projects", CLOUD_ARCHITECT_RESUME, JD_CLOUD_ARCHITECT, "projects", _contains_any("Terraform", "CloudFormation")),
    GoldenQuery("cloud_architect_education", CLOUD_ARCHITECT_RESUME, JD_CLOUD_ARCHITECT, "education", _contains_any("Computer Science")),
    # --- iOS / Tanvi ---
    GoldenQuery("ios_skills", IOS_RESUME, JD_IOS, "skills", _contains_any("Swift", "SwiftUI")),
    GoldenQuery("ios_experience", IOS_RESUME, JD_IOS, "experience", _contains_any("SwiftUI", "Core Data")),
    GoldenQuery("ios_projects", IOS_RESUME, JD_IOS, "projects", _contains_any("SwiftUI", "Core Data")),
    # --- Database Administration / Kabir ---
    GoldenQuery("dba_skills", DBA_RESUME, JD_DBA, "skills", _contains_any("PostgreSQL", "Database Tuning")),
    GoldenQuery("dba_experience", DBA_RESUME, JD_DBA, "experience", _contains_any("PostgreSQL", "index tuning")),
    GoldenQuery("dba_projects", DBA_RESUME, JD_DBA, "projects", _contains_any("PostgreSQL", "query performance")),
    GoldenQuery("dba_education", DBA_RESUME, JD_DBA, "education", _contains_any("Computer Science")),
    # --- Technical Support / DevRel / Ritika ---
    GoldenQuery("devrel_skills", DEVREL_RESUME, JD_DEVREL, "skills", _contains_any("Technical Writing", "REST APIs")),
    GoldenQuery("devrel_experience", DEVREL_RESUME, JD_DEVREL, "experience", _contains_any("documentation", "API reference")),
    GoldenQuery("devrel_projects", DEVREL_RESUME, JD_DEVREL, "projects", _contains_any("documentation", "REST API")),
    # --- Computer Vision / Aryan ---
    GoldenQuery("computer_vision_skills", COMPUTER_VISION_RESUME, JD_COMPUTER_VISION, "skills", _contains_any("OpenCV", "YOLO")),
    GoldenQuery("computer_vision_experience", COMPUTER_VISION_RESUME, JD_COMPUTER_VISION, "experience", _contains_any("OpenCV", "YOLO")),
    GoldenQuery("computer_vision_projects", COMPUTER_VISION_RESUME, JD_COMPUTER_VISION, "projects", _contains_any("OpenCV", "YOLO")),
    GoldenQuery("computer_vision_education", COMPUTER_VISION_RESUME, JD_COMPUTER_VISION, "education", _contains_any("Computer Science")),
    # --- Go/Rust Backend Systems / Devansh ---
    GoldenQuery("go_rust_skills", GO_RUST_RESUME, JD_GO_RUST_SYSTEMS, "skills", _contains_any("gRPC", "Rust")),
    GoldenQuery("go_rust_experience", GO_RUST_RESUME, JD_GO_RUST_SYSTEMS, "experience", _contains_any("gRPC", "Rust")),
    GoldenQuery("go_rust_projects", GO_RUST_RESUME, JD_GO_RUST_SYSTEMS, "projects", _contains_any("gRPC", "Rust")),
]

# Negative controls: resume x unrelated JD. Nothing in these resumes should
# be relevant to these JDs' requirements -- correct retrieval is empty.
NEGATIVE_CONTROLS: list[GoldenQuery] = [
    GoldenQuery("backend_resume_vs_ml_jd_skills", BACKEND_RESUME, JD_ML, "skills", lambda text: False),
    GoldenQuery("backend_resume_vs_ml_jd_experience", BACKEND_RESUME, JD_ML, "experience", lambda text: False),
    GoldenQuery("frontend_resume_vs_devops_jd_skills", FRONTEND_RESUME, JD_DEVOPS, "skills", lambda text: False),
    GoldenQuery("frontend_resume_vs_devops_jd_experience", FRONTEND_RESUME, JD_DEVOPS, "experience", lambda text: False),
    GoldenQuery("cybersecurity_resume_vs_frontend_jd_skills", CYBERSECURITY_RESUME, JD_FRONTEND, "skills", lambda text: False),
    GoldenQuery("cybersecurity_resume_vs_frontend_jd_experience", CYBERSECURITY_RESUME, JD_FRONTEND, "experience", lambda text: False),
    GoldenQuery("sre_resume_vs_ml_jd_skills", SRE_RESUME, JD_ML, "skills", lambda text: False),
    GoldenQuery("sre_resume_vs_ml_jd_experience", SRE_RESUME, JD_ML, "experience", lambda text: False),
    GoldenQuery("blockchain_resume_vs_qa_jd_skills", BLOCKCHAIN_RESUME, JD_QA, "skills", lambda text: False),
    GoldenQuery("blockchain_resume_vs_qa_jd_experience", BLOCKCHAIN_RESUME, JD_QA, "experience", lambda text: False),
    GoldenQuery("ios_resume_vs_embedded_jd_skills", IOS_RESUME, JD_EMBEDDED, "skills", lambda text: False),
    GoldenQuery("ios_resume_vs_embedded_jd_experience", IOS_RESUME, JD_EMBEDDED, "experience", lambda text: False),
    GoldenQuery("computer_vision_resume_vs_devops_jd_skills", COMPUTER_VISION_RESUME, JD_DEVOPS, "skills", lambda text: False),
    GoldenQuery("computer_vision_resume_vs_devops_jd_experience", COMPUTER_VISION_RESUME, JD_DEVOPS, "experience", lambda text: False),
]
