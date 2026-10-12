
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from contracts.models import DepthPlan, VendorProfile
from m2_web.parse import parse_page


def create_pdf():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(72, 720, "Security Report")
    pdf.drawString(72, 690, "We protect customer data.")
    pdf.save()
    return buffer.getvalue()


def test_parse_html():
    html = """
    <html>
      <head><title>Security Page</title></head>
      <body>
        <main>
          <h1>Security</h1>
          <p>We protect customer data.</p>
        </main>
      </body>
    </html>
    """

    title, text = parse_page(
        "https://example.com/security",
        html=html,
    )

    assert title == "Security Page"
    assert "Security" in text
    assert "customer data" in text


def test_vendor_profile():
    vendor = VendorProfile(
        vendor_id="example",
        name="Example",
        website="https://example.com",
    )

    assert vendor.vendor_id == "example"
    assert vendor.website == "https://example.com"


def test_depth_plan():
    plan = DepthPlan(vendor_id="example")

    assert plan.max_pages > 0
    assert plan.max_depth >= 0
    assert "TRUST" in plan.sources


def test_parse_pdf_text():
    pdf_bytes = create_pdf()

    title, text = parse_page(
        "https://example.com/security-report.pdf",
        pdf_bytes=pdf_bytes,
    )

    assert title == "security-report.pdf"
    assert "Security Report" in text
    assert "customer data" in text


def test_parse_invalid_pdf_safely():
    title, text = parse_page(
        "https://example.com/broken.pdf",
        pdf_bytes=b"not a real PDF",
    )

    assert title == "broken.pdf"
    assert text == ""


def test_pdf_table_extraction():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    x_positions = [72, 260, 440]
    y_positions = [720, 690, 660]

    for y in y_positions:
        pdf.line(x_positions[0], y, x_positions[-1], y)

    for x in x_positions:
        pdf.line(x, y_positions[0], x, y_positions[-1])

    pdf.drawString(80, 700, "Vendor")
    pdf.drawString(268, 700, "Service")
    pdf.drawString(80, 670, "Example AI")
    pdf.drawString(268, 670, "Data processing")
    pdf.save()

    pdf_bytes = buffer.getvalue()
    title, text = parse_page(
        "https://example.com/vendors.pdf",
        pdf_bytes=pdf_bytes,
    )

    assert title == "vendors.pdf"
    assert "Example AI" in text
    assert "Data processing" in text
