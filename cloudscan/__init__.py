"""awscloudscan — a CloudFox + PMapper wrapper with DeepSeek-powered analysis.

Awscloudscan runs CloudFox and PMapper against a single AWS account (selected by an
AWS CLI profile), has DeepSeek review the combined output, derives the possible
privilege-escalation paths, and writes a Markdown report scoped to the account
itself (excluding anything that belongs to a parent AWS Organization).
"""

__version__ = "0.1.0"
