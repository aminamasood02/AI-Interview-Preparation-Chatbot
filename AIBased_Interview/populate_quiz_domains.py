#!/usr/bin/env python
"""
Script to populate initial quiz domains
Run this after migrations to set up quiz domains
"""

import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AIBased_Interview.settings')
django.setup()

from AIBased_Interview.models import QuizDomain

def populate_domains():
    """Populate initial quiz domains"""
    domains_data = [
        {
            'name': 'python',
            'display_name': 'Python Programming',
            'description': 'Test your Python programming skills with questions on syntax, data structures, OOP, and more.',
            'icon': 'fab fa-python'
        },
        {
            'name': 'javascript',
            'display_name': 'JavaScript',
            'description': 'Assess your JavaScript knowledge including ES6+, DOM manipulation, and modern frameworks.',
            'icon': 'fab fa-js-square'
        },
        {
            'name': 'java',
            'display_name': 'Java Programming',
            'description': 'Evaluate your Java skills covering OOP, collections, multithreading, and Spring framework.',
            'icon': 'fab fa-java'
        },
        {
            'name': 'react',
            'display_name': 'React.js',
            'description': 'Test your React knowledge including hooks, state management, and component lifecycle.',
            'icon': 'fab fa-react'
        },
        {
            'name': 'django',
            'display_name': 'Django Framework',
            'description': 'Assess your Django skills including models, views, templates, and REST APIs.',
            'icon': 'fas fa-server'
        },
        {
            'name': 'data_science',
            'display_name': 'Data Science',
            'description': 'Test your data science knowledge including statistics, pandas, numpy, and machine learning.',
            'icon': 'fas fa-chart-line'
        },
        {
            'name': 'machine_learning',
            'display_name': 'Machine Learning',
            'description': 'Evaluate your ML skills including algorithms, model selection, and deep learning concepts.',
            'icon': 'fas fa-brain'
        },
        {
            'name': 'web_development',
            'display_name': 'Web Development',
            'description': 'Test your web development skills including HTML, CSS, responsive design, and best practices.',
            'icon': 'fas fa-globe'
        },
        {
            'name': 'database',
            'display_name': 'Database Management',
            'description': 'Assess your database knowledge including SQL, NoSQL, normalization, and query optimization.',
            'icon': 'fas fa-database'
        },
        {
            'name': 'algorithms',
            'display_name': 'Algorithms & Data Structures',
            'description': 'Test your algorithmic thinking and knowledge of data structures, complexity analysis.',
            'icon': 'fas fa-project-diagram'
        },
        {
            'name': 'system_design',
            'display_name': 'System Design',
            'description': 'Evaluate your system design skills including scalability, architecture patterns, and trade-offs.',
            'icon': 'fas fa-sitemap'
        },
        {
            'name': 'networking',
            'display_name': 'Computer Networks',
            'description': 'Test your networking knowledge including protocols, OSI model, and network security.',
            'icon': 'fas fa-network-wired'
        },
        {
            'name': 'cybersecurity',
            'display_name': 'Cybersecurity',
            'description': 'Assess your cybersecurity knowledge including threats, encryption, and security best practices.',
            'icon': 'fas fa-shield-alt'
        },
        {
            'name': 'cloud_computing',
            'display_name': 'Cloud Computing',
            'description': 'Test your cloud knowledge including AWS, Azure, containerization, and microservices.',
            'icon': 'fas fa-cloud'
        },
        {
            'name': 'devops',
            'display_name': 'DevOps',
            'description': 'Evaluate your DevOps skills including CI/CD, automation, monitoring, and infrastructure.',
            'icon': 'fas fa-tools'
        }
    ]
    
    created_count = 0
    for domain_data in domains_data:
        domain, created = QuizDomain.objects.get_or_create(
            name=domain_data['name'],
            defaults=domain_data
        )
        if created:
            created_count += 1
            print(f"Created domain: {domain.display_name}")
        else:
            print(f"Domain already exists: {domain.display_name}")
    
    print(f"\nPopulation complete! Created {created_count} new domains.")
    print(f"Total domains in database: {QuizDomain.objects.count()}")

if __name__ == '__main__':
    populate_domains() 