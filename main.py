from fastapi import FastAPI, HTTPException, Path, Query, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
import httpx
import json
import os
from datetime import datetime, timedelta
import math
import re

# Initialize FastAPI
app = FastAPI(
    title="GitHub Reputation Kernel",
    description="An off-chain kernel for analyzing GitHub contributions and generating reputation metrics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup GitHub API integration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Store your GitHub API token in environment variables

# Models
class ContributionMetrics(BaseModel):
    total_commits: int = Field(..., description="Total number of commits across all repositories")
    total_prs: int = Field(..., description="Total number of pull requests")
    total_issues: int = Field(..., description="Total number of issues opened")
    total_stars: int = Field(..., description="Total stars on repositories contributed to")
    contribution_streak: int = Field(..., description="Longest streak of consecutive days with contributions")
    languages: Dict[str, int] = Field(..., description="Programming languages used and line counts")
    repositories_contributed: int = Field(..., description="Number of repositories contributed to")
    account_age_days: int = Field(..., description="Age of GitHub account in days")
    activity_consistency: float = Field(..., description="Consistency of activity score (0-100)")

class ReputationScore(BaseModel):
    overall_score: int = Field(..., description="Overall reputation score (0-100)")
    experience_level: str = Field(..., description="Developer experience level category")
    contribution_level: str = Field(..., description="Contribution activity level")
    specialty_areas: List[str] = Field(..., description="Areas of specialty based on repositories and languages")
    badge_tier: str = Field(..., description="NFT badge tier eligibility (Bronze, Silver, Gold, Platinum)")
    badge_attributes: Dict[str, Any] = Field(..., description="Special attributes for the NFT badge")

class GitHubReputationResponse(BaseModel):
    github_username: str = Field(..., description="GitHub username analyzed")
    metrics: ContributionMetrics = Field(..., description="Detailed contribution metrics")
    reputation: ReputationScore = Field(..., description="Reputation score and badge eligibility")
    last_updated: str = Field(..., description="Timestamp when this data was generated")
    avatar_url: Optional[str] = Field(None, description="GitHub avatar URL")

class GitHubRequest(BaseModel):
    github_username: str = Field(..., description="GitHub username to analyze")
    wallet_address: str = Field(..., description="Ethereum wallet address for verification")
    
    @field_validator('github_username')
    def validate_github_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9]*$', v):
            raise ValueError('Invalid GitHub username format')
        return v
    
    @field_validator('wallet_address')
    def validate_wallet_address(cls, v):
        if not re.match(r'^0x[a-fA-F0-9]{40}$', v):
            raise ValueError('Invalid Ethereum wallet address format')
        return v

# GitHub API client
async def github_client():
    async with httpx.AsyncClient(base_url=GITHUB_API_BASE, timeout=30.0) as client:
        if GITHUB_TOKEN:
            client.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
        yield client

# Helper Functions
async def get_user_profile(client, username):
    response = await client.get(f"/users/{username}")
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail=f"GitHub user {username} not found")
    return response.json()

async def get_user_repos(client, username):
    # Get all repositories (including forks)
    response = await client.get(f"/users/{username}/repos", params={"per_page": 100})
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Could not fetch repositories for {username}")
    return response.json()

async def get_user_commits(client, username):
    # This is a simplified version, in reality you'd need to handle pagination and get commits across repositories
    response = await client.get(f"/search/commits", 
                               params={"q": f"author:{username}", "per_page": 100},
                               headers={"Accept": "application/vnd.github.cloak-preview"})
    if response.status_code != 200:
        return []  # Return empty list rather than failing
    return response.json().get("items", [])

async def get_user_prs(client, username):
    response = await client.get(f"/search/issues", 
                               params={"q": f"author:{username} type:pr", "per_page": 100})
    if response.status_code != 200:
        return []
    return response.json().get("items", [])

async def get_user_issues(client, username):
    response = await client.get(f"/search/issues", 
                               params={"q": f"author:{username} type:issue", "per_page": 100})
    if response.status_code != 200:
        return []
    return response.json().get("items", [])

async def get_contribution_stats(client, username):
    # In a real implementation, you'd handle the events endpoint pagination
    # and analyze the contribution pattern over time
    response = await client.get(f"/users/{username}/events", params={"per_page": 100})
    if response.status_code != 200:
        return {}
    events = response.json()
    
    # Calculate contribution streak and consistency
    # This is simplified - a real implementation would be more complex
    dates = {}
    for event in events:
        event_date = event.get("created_at", "").split("T")[0]
        if event_date:
            dates[event_date] = dates.get(event_date, 0) + 1
    
    # Calculate streak (simplified)
    sorted_dates = sorted(dates.keys())
    current_streak = 1
    max_streak = 1
    
    for i in range(1, len(sorted_dates)):
        d1 = datetime.strptime(sorted_dates[i-1], "%Y-%m-%d")
        d2 = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
        if (d2 - d1).days == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    
    # Calculate consistency (percentage of days with activity in last 90 days)
    consistency = len(dates) / 90.0 * 100 if dates else 0
    consistency = min(100, consistency)  # Cap at 100%
    
    return {
        "streak": max_streak,
        "consistency": consistency
    }

async def calculate_reputation(metrics):
    # Convert metrics into a reputation score (0-100)
    # This is a simplified algorithm - you would implement a more sophisticated one
    
    # Base score from commits and PRs
    commit_factor = min(100, metrics.total_commits / 5)  # Cap at 100
    pr_factor = min(100, metrics.total_prs * 2)  # Cap at 100
    
    # Activity factors
    streak_factor = min(100, metrics.contribution_streak * 2)  # Cap at 100
    consistency_factor = metrics.activity_consistency
    
    # Experience factors
    repo_diversity = min(100, metrics.repositories_contributed * 5)  # Cap at 100
    account_age_factor = min(100, metrics.account_age_days / 30)  # Cap at 100
    
    # Calculate overall score with weighted factors
    overall_score = int((
        commit_factor * 0.25 +
        pr_factor * 0.25 +
        streak_factor * 0.15 +
        consistency_factor * 0.15 +
        repo_diversity * 0.1 +
        account_age_factor * 0.1
    ))
    
    # Determine experience level
    if overall_score >= 85:
        experience_level = "Expert"
    elif overall_score >= 70:
        experience_level = "Advanced"
    elif overall_score >= 40:
        experience_level = "Intermediate"
    else:
        experience_level = "Beginner"
    
    # Determine contribution level
    if metrics.total_commits > 500 or metrics.total_prs > 100:
        contribution_level = "Very Active"
    elif metrics.total_commits > 200 or metrics.total_prs > 50:
        contribution_level = "Active"
    elif metrics.total_commits > 50 or metrics.total_prs > 10:
        contribution_level = "Moderate"
    else:
        contribution_level = "Occasional"
    
    # Determine badge tier
    if overall_score >= 85:
        badge_tier = "Platinum"
    elif overall_score >= 70:
        badge_tier = "Gold"
    elif overall_score >= 50:
        badge_tier = "Silver"
    else:
        badge_tier = "Bronze"
    
    # Determine specialty areas based on languages and repositories
    # This is simplified - you would implement a more sophisticated analysis
    specialty_areas = []
    top_languages = sorted(metrics.languages.items(), key=lambda x: x[1], reverse=True)[:3]
    specialty_areas = [lang for lang, _ in top_languages]
    
    # Define special attributes for the NFT
    badge_attributes = {
        "tier": badge_tier,
        "commit_count": metrics.total_commits,
        "pr_count": metrics.total_prs,
        "languages": top_languages,
        "streak": metrics.contribution_streak,
        "experience": experience_level
    }
    
    return ReputationScore(
        overall_score=overall_score,
        experience_level=experience_level,
        contribution_level=contribution_level,
        specialty_areas=specialty_areas,
        badge_tier=badge_tier,
        badge_attributes=badge_attributes
    )

# API Endpoints

@app.get("/reputation/{github_username}", response_model=GitHubReputationResponse)
async def get_github_reputation(
    github_username: str = Path(..., description="GitHub username to analyze"),
    wallet_address: str = Query(..., description="Ethereum wallet address for verification"),
    client: httpx.AsyncClient = Depends(github_client)
):
    """
    Get the GitHub reputation score and badge eligibility for a user.
    This endpoint analyzes GitHub contribution history and provides metrics for NFT badges.
    """
    # Validate the input
    if not re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9]*$', github_username):
        raise HTTPException(status_code=400, detail="Invalid GitHub username format")
    
    if not re.match(r'^0x[a-fA-F0-9]{40}$', wallet_address):
        raise HTTPException(status_code=400, detail="Invalid Ethereum wallet address format")
    
    try:
        # Get user profile
        profile = await get_user_profile(client, github_username)
        
        # Get repositories
        repos = await get_user_repos(client, github_username)
        
        # Get commits, PRs, and issues
        commits = await get_user_commits(client, github_username)
        prs = await get_user_prs(client, github_username)
        issues = await get_user_issues(client, github_username)
        
        # Get contribution stats
        contribution_stats = await get_contribution_stats(client, github_username)
        
        # Calculate account age
        created_at = datetime.strptime(profile.get("created_at"), "%Y-%m-%dT%H:%M:%SZ")
        account_age_days = (datetime.now() - created_at).days
        
        # Calculate languages used (simplified)
        languages = {}
        for repo in repos:
            if repo.get("language") and not repo.get("fork"):
                lang = repo.get("language")
                languages[lang] = languages.get(lang, 0) + 1
        
        # Compile metrics
        metrics = ContributionMetrics(
            total_commits=len(commits),
            total_prs=len(prs),
            total_issues=len(issues),
            total_stars=sum(repo.get("stargazers_count", 0) for repo in repos if not repo.get("fork")),
            contribution_streak=contribution_stats.get("streak", 0),
            languages=languages,
            repositories_contributed=len([r for r in repos if not r.get("fork")]),
            account_age_days=account_age_days,
            activity_consistency=contribution_stats.get("consistency", 0)
        )
        
        # Calculate reputation score
        reputation = await calculate_reputation(metrics)
        
        # Prepare response
        response = GitHubReputationResponse(
            github_username=github_username,
            metrics=metrics,
            reputation=reputation,
            last_updated=datetime.now().isoformat(),
            avatar_url=profile.get("avatar_url")
        )
        
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error analyzing GitHub data: {str(e)}")

@app.post("/reputation", response_model=GitHubReputationResponse)
async def post_github_reputation(
    request: GitHubRequest,
    client: httpx.AsyncClient = Depends(github_client)
):
    """
    Post-based endpoint for GitHub reputation analysis.
    Functionally identical to the GET endpoint but accepts POST request with JSON body.
    """
    return await get_github_reputation(
        github_username=request.github_username,
        wallet_address=request.wallet_address,
        client=client
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)