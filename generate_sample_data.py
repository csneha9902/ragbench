import os
import sys

def check_and_install_reportlab():
    try:
        from reportlab.lib.pagesizes import letter
    except ImportError:
        print("ReportLab is not installed yet. Please wait for dependencies to finish installing.")
        sys.exit(1)

def main():
    check_and_install_reportlab()
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    def create_pdf(filename, title, content_paragraphs):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontSize=22,
            leading=26,
            spaceAfter=20,
            textColor='#1E3A8A' # Deep Blue
        )
        heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor='#0D9488' # Teal
        )
        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['BodyText'],
            fontSize=10,
            leading=14,
            spaceAfter=10
        )
        
        story = [Paragraph(title, title_style), Spacer(1, 12)]
        
        for item in content_paragraphs:
            if item.startswith("## "):
                story.append(Paragraph(item[3:], heading_style))
            else:
                story.append(Paragraph(item, body_style))
                
        doc.build(story)
        print(f"Created PDF: {filename}")

    # 1. Leave Policy Handbook
    leave_content = [
        "This document outlines the official leave policies of RAGBench Enterprises. We believe in providing our employees with sufficient time to rest, recover, and attend to personal matters.",
        "## Section 1: Annual Leave (Vacation)",
        "All full-time employees accrue paid annual leave (vacation time) at a rate of 1.25 days per month, which totals exactly 15 annual leave days per calendar year. Vacation time must be requested at least two weeks in advance via the HR portal and approved by the direct manager. A maximum of 5 unused vacation days can be carried over to the next calendar year; any additional unused days beyond 5 will be forfeited on December 31st.",
        "## Section 2: Paid Sick Leave",
        "Employees receive 10 paid sick days per calendar year, allotted on January 1st. Sick leave is designed for personal illness, medical appointments, or caring for an immediate family member who is ill. If an employee is absent for more than three consecutive sick days, a medical certificate from a licensed physician must be submitted to the HR department.",
        "## Section 3: Parental Leave",
        "RAGBench Enterprises offers comprehensive parental leave support. Birthing mothers are entitled to 26 weeks of fully paid maternity leave to support recovery and bonding. Non-birthing parents, including adoptive and foster parents, are entitled to 4 weeks of fully paid paternity/parental leave. All parental leave must be taken within the first 12 months of the child's birth or placement.",
        "## Section 4: Bereavement Leave",
        "In the unfortunate event of the death of an immediate family member (spouse, child, parent, or sibling), employees are eligible for up to 5 consecutive paid days off for bereavement, funeral attendance, and estate planning. For extended family members (grandparents, aunts, uncles), 2 paid days off are provided.",
        "## Section 5: Public Holidays",
        "The company officially observes 11 public holidays per year, during which offices are closed and employees receive full pay. These include New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving Day, Christmas Day, and others listed in the annual company calendar."
    ]
    create_pdf("data/leave_policy.pdf", "RAGBench Enterprises - Employee Leave Policy Handbook", leave_content)

    # 2. Employee Benefits Guide
    benefits_content = [
        "Welcome to the RAGBench Enterprises Employee Benefits Guide. This document summarizes the benefits and perks available to full-time employees.",
        "## Section 1: Health, Dental, and Vision Insurance",
        "We partner with Blue Cross to offer comprehensive health insurance coverage. The company covers 80% of the premium for employees and their dependents, while the remaining 20% is deducted pre-tax from the employee's salary. Dental coverage is provided through Delta Dental, and vision coverage is provided through VSP. Both dental and vision premiums are covered 100% by the company for the employee.",
        "## Section 2: 401(k) Retirement Savings Match",
        "Employees are eligible to participate in the company 401(k) plan after completing 3 months of continuous employment. RAGBench Enterprises will match 100% of employee contributions up to a maximum of 4% of the employee's base salary. The matched contributions are subject to a 2-year vesting schedule, where 50% vests after 1 year, and 100% vests after 2 years.",
        "## Section 3: Health Savings Account (HSA)",
        "For employees who enroll in our High Deductible Health Plan (HDHP), the company offers a Health Savings Account (HSA) option. The company will contribute $500 annually for individual coverage and $1,000 annually for family coverage, deposited in quarterly installments. Employees may also make pre-tax contributions up to the IRS legal limit.",
        "## Section 4: Gym & Wellness Reimbursement",
        "To promote physical health, the company provides a wellness stipend of up to $50 per month. This stipend can be used to reimburse gym memberships, fitness classes, yoga sessions, or health app subscriptions. Claims must be submitted via expense reports with receipts by the end of each calendar quarter.",
        "## Section 5: Tuition Assistance Program",
        "RAGBench Enterprises supports the continuous professional development of our employees. We offer tuition assistance of up to $5,250 per calendar year for approved, job-related courses, certifications, or degree programs. To qualify, the course must be pre-approved by the manager and HR, and the employee must achieve a grade of 'B' or higher (or 'Pass' in pass/fail courses) to receive reimbursement."
    ]
    create_pdf("data/employee_benefits.pdf", "RAGBench Enterprises - Employee Benefits & Perks Guide", benefits_content)

    # 3. Code of Conduct & Remote Work Policy
    conduct_content = [
        "This document defines the code of conduct and remote work policies that govern all employees of RAGBench Enterprises.",
        "## Section 1: Workplace Professionalism & Diversity",
        "We are committed to maintaining a safe, respectful, and inclusive work environment. The company enforces a zero-tolerance policy for harassment, discrimination, or bullying of any kind, whether based on race, gender, religion, sexual orientation, age, or disability. Violations will result in immediate disciplinary action up to and including termination.",
        "## Section 2: Conflicts of Interest",
        "Employees must avoid any interest, influence, or relationship that conflicts or appears to conflict with the best interests of RAGBench Enterprises. This includes accepting valuable gifts (over $100 value) from clients/vendors, or engaging in external employment that competes directly with the company or interferes with the employee's regular duties.",
        "## Section 3: Data Security & Privacy",
        "Protecting company and client data is a critical responsibility. All employees must follow password guidelines (minimum 12 characters, mixing letters, numbers, and symbols) and use Multi-Factor Authentication (MFA) on all business accounts. Business-related tasks should only be conducted on company-provided, secure laptops, which must be locked when unattended.",
        "## Section 4: Flexible Remote Work Policy",
        "RAGBench Enterprises supports a hybrid and flexible work environment. Employees may work remotely up to 4 days per week, with Wednesday designated as the mandatory in-office collaboration day. Core working hours are 10:00 AM to 4:00 PM EST, during which all employees are expected to be online, responsive on Slack, and available for meetings. The company provides a remote work internet subsidy of $40 per month, reimbursable through quarterly expenses.",
        "## Section 5: Equipment Usage & Return Policy",
        "All hardware and software tools provided by the company remain company property. Laptops, monitors, and accessories are intended solely for professional business use. Upon termination of employment, all company-owned hardware must be returned to the office or shipped back using prepaid shipping labels within 5 business days."
    ]
    create_pdf("data/code_of_conduct.pdf", "RAGBench Enterprises - Code of Conduct & Remote Work Policy", conduct_content)

if __name__ == "__main__":
    main()
