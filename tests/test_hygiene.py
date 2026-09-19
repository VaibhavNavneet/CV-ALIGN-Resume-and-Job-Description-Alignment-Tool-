"""Tests for the deterministic hygiene checker."""

from __future__ import annotations

from multi_agent_resume_screener.pipeline.hygiene import check_hygiene
from multi_agent_resume_screener.state import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    StructuredResume,
)


def _strong_resume() -> StructuredResume:
    return StructuredResume(
        name="Jane Smith",
        email="jane@example-mail.org",
        skills=["Python", "Go", "Docker", "Kubernetes", "PostgreSQL"],
        experience=[
            ExperienceItem(
                company="Acme",
                role="SWE Intern",
                bullets=["Reduced latency by 40% serving 10k requests/day"],
            )
        ],
        projects=[ProjectItem(name="RAG Knowledge Assistant", tech=["Python"])],
        education=[EducationItem(degree="B.Tech CS", institute="IIT")],
    )


def _strong_raw_text() -> str:
    return (
        "Jane Smith\njane@example-mail.org\n"
        "https://github.com/jane  https://linkedin.com/in/jane\n"
    )


def test_strong_resume_scores_high_with_no_warnings():
    report = check_hygiene(_strong_resume(), _strong_raw_text())
    warnings = [i for i in report.issues if i.severity == "warning"]
    assert warnings == []
    assert report.score >= 0.9
    assert report.positives  # collected some positives


def test_missing_links_flagged():
    report = check_hygiene(_strong_resume(), raw_text="no links here")
    checks = {i.check for i in report.issues}
    assert "missing_github" in checks
    assert "no_links" in checks


def test_thin_skills_flagged():
    resume = _strong_resume()
    resume.skills = ["Python"]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "thin_skills" for i in report.issues)


def test_no_projects_flagged_as_warning():
    resume = _strong_resume()
    resume.projects = []
    report = check_hygiene(resume, _strong_raw_text())
    assert any(
        i.check == "no_projects" and i.severity == "warning"
        for i in report.issues
    )


def test_generic_project_name_flagged():
    resume = _strong_resume()
    resume.projects = [ProjectItem(name="Project 1")]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "generic_project_name" for i in report.issues)


def test_unquantified_experience_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(company="Acme", role="Intern",
                       bullets=["Worked on backend services"])
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "unquantified_experience" for i in report.issues)


def test_placeholder_url_flagged():
    resume = _strong_resume()
    report = check_hygiene(resume, raw_text="see https://github.com/x and example.com")
    assert any(i.check == "placeholder_url" for i in report.issues)


def test_bare_domain_links_not_flagged_as_no_links():
    # Links without an http(s):// prefix (e.g. "github.com/x") still count as
    # links; "no_links" must not fire when GitHub/LinkedIn are present.
    resume = _strong_resume()
    report = check_hygiene(resume, raw_text="github.com/jane linkedin.com/in/jane")
    checks = {i.check for i in report.issues}
    assert "no_links" not in checks
    assert "missing_github" not in checks


def test_score_never_negative():
    # An almost-empty resume accrues many penalties but score stays >= 0.
    report = check_hygiene(StructuredResume(), raw_text="")
    assert 0.0 <= report.score <= 1.0


def test_deterministic():
    resume = _strong_resume()
    raw = _strong_raw_text()
    assert check_hygiene(resume, raw).score == check_hygiene(resume, raw).score


def test_weak_action_verb_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["Responsible for maintaining the deployment pipeline in 2023"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "weak_action_verb" for i in report.issues)


def test_strong_verb_not_flagged_as_weak():
    # The strong resume's bullet starts with "Reduced" and must not be flagged.
    report = check_hygiene(_strong_resume(), _strong_raw_text())
    assert not any(i.check == "weak_action_verb" for i in report.issues)


def test_long_bullet_flagged():
    resume = _strong_resume()
    long_bullet = "Built " + " ".join(f"thing{i}" for i in range(40)) + " achieving 10%"
    resume.experience = [
        ExperienceItem(company="Acme", role="Intern", bullets=[long_bullet])
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "long_bullet" for i in report.issues)


def test_first_person_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["Built my own service that cut costs by 20%"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "first_person" for i in report.issues)


def test_first_person_ignores_ie_abbreviation():
    # "i.e." must not be mistaken for the pronoun "I".
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["Reduced latency, i.e. faster responses, by 40% across 3 apps"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert not any(i.check == "first_person" for i in report.issues)


def test_buzzwords_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["A hardworking team player who reduced latency by 40%"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "buzzwords" for i in report.issues)


def test_strong_resume_triggers_no_quickfix_rules():
    # None of the recruiter quick-fix rules should fire on a clean resume.
    report = check_hygiene(_strong_resume(), _strong_raw_text())
    quickfix = {"weak_action_verb", "long_bullet", "first_person", "buzzwords"}
    assert not any(i.check in quickfix for i in report.issues)
