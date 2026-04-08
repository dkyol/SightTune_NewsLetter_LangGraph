import base64
from datetime import date
from pathlib import Path

LOGO_PATH = Path(__file__).parent.parent / "assets" / "logo.png"


def load_logo_b64() -> str:
    if not LOGO_PATH.exists():
        raise FileNotFoundError(f"Logo not found at {LOGO_PATH} — add assets/logo.png before running")
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


def parse_article(text: str) -> tuple[str, str]:
    """Split article text into (headline, body) — strips markdown bold/headers."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    headline = lines[0].strip().strip("#*").strip() if lines else "Untitled"
    body = " ".join(l.strip() for l in lines[1:]).strip()
    return headline, body


def build_email(theme: str, articles: list, today: str, logo_b64: str) -> str:
    articles_html = ""
    for i, article in enumerate(articles, 1):
        headline, body = parse_article(article)
        divider = (
            '<tr><td style="padding:0 48px;">'
            '<table width="100%" cellpadding="0" cellspacing="0">'
            '<tr><td style="border-top:1px solid #e8e8e4;"></td></tr>'
            "</table></td></tr>"
        ) if i > 1 else ""
        articles_html += f"""
        {divider}
        <tr>
          <td align="center" style="padding:40px 48px 36px 48px;text-align:center;">
            <h2 style="margin:0 0 16px 0;font-size:20px;font-weight:normal;color:#1a1a1a;
                       font-family:Georgia,'Times New Roman',serif;line-height:1.4;
                       text-align:center;">{headline}</h2>
            <p style="margin:0;font-size:15px;line-height:1.8;color:#555555;
                      font-family:Arial,Helvetica,sans-serif;text-align:center;">{body}</p>
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>SightTune Newsletter — {today}</title>
</head>
<body style="margin:0;padding:0;background-color:#f5f5f3;-webkit-font-smoothing:antialiased;">

<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background-color:#f5f5f3;">
  <tr>
    <td align="center" style="padding:48px 16px 64px 16px;">

      <table width="600" cellpadding="0" cellspacing="0" border="0"
             style="max-width:600px;width:100%;background-color:#ffffff;
                    box-shadow:0 2px 24px rgba(0,0,0,0.06);">

        <!-- HEADER -->
        <tr>
          <td align="center" style="padding:56px 48px 44px 48px;background-color:#ffffff;">
            <img src="data:image/png;base64,{logo_b64}"
                 width="260" alt="SightTune"
                 style="display:block;margin:0 auto;"/>
            <p style="margin:16px 0 5px 0;font-size:9px;letter-spacing:5px;color:#c0c0c0;
                      text-transform:uppercase;font-family:Arial,Helvetica,sans-serif;">Music Technology</p>
            <p style="margin:0;font-size:11px;letter-spacing:1px;color:#d0d0d0;
                      font-family:Arial,Helvetica,sans-serif;">{today}</p>
          </td>
        </tr>

        <!-- TOP RULE -->
        <tr>
          <td style="padding:0 48px;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr><td style="border-top:2px solid #1a1a1a;"></td></tr>
            </table>
          </td>
        </tr>

        <!-- ISSUE LABEL -->
        <tr>
          <td align="center" style="padding:28px 48px 0 48px;">
            <p style="margin:0;font-size:15px;font-weight:bold;color:#1a1a1a;
                      font-family:Arial,Helvetica,sans-serif;letter-spacing:1px;">SightTune</p>
          </td>
        </tr>

        <!-- ARTICLES -->
        {articles_html}

        <!-- BOTTOM RULE -->
        <tr>
          <td style="padding:0 48px;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr><td style="border-top:2px solid #1a1a1a;"></td></tr>
            </table>
          </td>
        </tr>

        <!-- MAILING LIST CTA -->
        <tr>
          <td align="center" style="padding:44px 48px;background-color:#f9f9f7;">
            <p style="margin:0 0 6px 0;font-size:9px;letter-spacing:4px;color:#aaaaaa;
                      text-transform:uppercase;font-family:Arial,Helvetica,sans-serif;">Stay in the loop</p>
            <p style="margin:0 auto 24px auto;font-size:14px;line-height:1.75;color:#888888;
                      font-family:Arial,Helvetica,sans-serif;max-width:340px;text-align:center;">
              Know a pianist who'd love this newsletter?
              Forward this email or share the sign-up link below.
            </p>
            <a href="https://forms.gle/YOUR_GOOGLE_FORM_ID"
               style="display:inline-block;background-color:#ffffff;color:#1a1a1a;
                      text-decoration:none;padding:14px 36px;border:1px solid #1a1a1a;
                      font-size:9px;letter-spacing:3px;text-transform:uppercase;
                      font-family:Arial,Helvetica,sans-serif;">
              Subscribe to Newsletter
            </a>
          </td>
        </tr>

        <!-- TESTFLIGHT CTA -->
        <tr>
          <td align="center" style="padding:52px 48px 56px 48px;background-color:#ffffff;">
            <img src="data:image/png;base64,{logo_b64}"
                 width="144" alt="SightTune"
                 style="display:block;margin:0 auto;"/>
            <p style="margin:24px 0 6px 0;font-size:9px;letter-spacing:4px;color:#aaaaaa;
                      text-transform:uppercase;font-family:Arial,Helvetica,sans-serif;">Now in Beta</p>
            <p style="margin:0 auto 32px auto;font-size:14px;line-height:1.75;color:#888888;
                      font-family:Arial,Helvetica,sans-serif;max-width:340px;text-align:center;">
              Experience AI-powered computer vision for pianists.
              Join the SightTune beta on TestFlight today.
            </p>
            <a href="https://testflight.apple.com/join/Wu5Vs9q4"
               style="display:inline-block;background-color:#1a1a1a;color:#ffffff;
                      text-decoration:none;padding:16px 40px;
                      font-size:9px;letter-spacing:3px;text-transform:uppercase;
                      font-family:Arial,Helvetica,sans-serif;">
              Join Beta on TestFlight
            </a>
          </td>
        </tr>

        <!-- FOOTER -->
        <tr>
          <td align="center"
              style="padding:20px 48px 32px 48px;border-top:1px solid #f0f0ee;
                     background-color:#ffffff;">
            <p style="margin:0;font-size:11px;color:#cccccc;
                      font-family:Arial,Helvetica,sans-serif;letter-spacing:0.5px;">
              &copy; {date.today().year} SightTune Music Technology &nbsp;&middot;&nbsp;
              <a href="https://www.sighttune.com"
                 style="color:#cccccc;text-decoration:none;">SightTune.com</a>
            </p>
          </td>
        </tr>

      </table>
    </td>
  </tr>
</table>
</body>
</html>"""
