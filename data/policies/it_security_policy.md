---
title: IT Security Policy
policy_id: "SEC-001"
version: "5.2"
effective_date: 2025-10-01
owner: "Information Security"
---
# IT Security Policy

## SEC-1 Purpose and Scope

This policy defines the minimum security practices required to protect Halvorsen Analytics' systems, data, and customers. It applies to all employees, contractors, and third parties who access company systems or data, whether in the office, at home, or while traveling. Violations may result in disciplinary action, up to and including termination.

## SEC-2 Data Classification

All company information is classified into one of four levels:

- **Public:** approved for external release.
- **Internal:** for employees only; low impact if disclosed.
- **Confidential:** customer data, financial results, and unreleased product plans.
- **Restricted:** credentials, encryption keys, and personal data covered by regulation.

Confidential and Restricted data must never be stored on personal devices, personal cloud storage, or removable media. Restricted data must be encrypted both at rest and in transit.

## SEC-3 Passwords and Authentication

Passwords must be at least **14 characters** long and must not be reused across systems. All employees must use the company password manager. **Multi-factor authentication (MFA)** is mandatory for email, the VPN, the HR portal, the expense system, and all production systems. Hardware security keys are required for administrators of production systems. Passwords must never be shared, including with IT staff.

## SEC-4 Devices

Company data may only be accessed from company-managed laptops and phones enrolled in device management. Devices must have full-disk encryption, automatic updates enabled, and a screen lock that activates after **5 minutes of inactivity**. Security patches classified as critical must be installed within 72 hours of release. Lost or stolen devices must be reported to the IT Service Desk immediately so they can be wiped remotely.

## SEC-5 Network Access

The company VPN must be used whenever an employee accesses internal systems from any network other than the office network or their own home network, including hotels, airports, and cafés. Public Wi-Fi must never be used without the VPN. Personal devices may connect only to the guest Wi-Fi in the office.

## SEC-6 Acceptable Use

Company systems are provided for business purposes; limited personal use is allowed if it does not interfere with work or create risk. Installing unapproved software is prohibited; requests for new software go through the IT Service Desk. Company data must not be pasted into public AI tools or external services that are not on the approved software list.

## SEC-7 Security Incidents

### SEC-7.1 What Counts as an Incident

A security incident is any event that could compromise the confidentiality, integrity, or availability of company data or systems. Examples include phishing emails that were clicked, malware alerts, lost devices, unexpected MFA prompts, and data sent to the wrong recipient.

### SEC-7.2 Reporting

All suspected incidents must be reported within **1 hour** of discovery to the Security Operations Center using the **#sec-incident** channel or the incident hotline. Do not attempt to investigate on your own, and do not delete suspicious emails until Security has reviewed them.

### SEC-7.3 Incident Escalation

Security Operations classifies each incident as **Severity 1 to Severity 4**. Severity 1 incidents, such as confirmed breaches of Restricted data, are escalated to the Chief Information Security Officer within 15 minutes and to the executive team within 1 hour. Severity 2 incidents are escalated to the Head of Security within 4 hours. Incidents involving personal data are referred to Legal to assess regulatory notification deadlines.

## SEC-8 Access Reviews

Managers must review their team's system access every quarter and remove access that is no longer needed. Access for departing employees is revoked on their last working day. Shared accounts are not permitted except for documented service accounts.

## SEC-9 Training

All employees must complete security awareness training within 14 days of joining and annually thereafter. Employees who fail a simulated phishing test must complete a short refresher module within 7 days.
