/**
 * SightTune Newsletter — Google Apps Script
 *
 * Paste into: Google Form → Extensions → Apps Script
 * Trigger: onFormSubmit → From form → On form submit
 *
 * Google Forms automatically writes responses to the linked Sheet.
 * This script runs AFTER that write and handles:
 *   - Duplicate removal
 *   - 2000-subscriber cap enforcement
 *
 * Setup:
 *   1. In your Google Form click Responses → Link to Sheets → Create new sheet
 *   2. Open that Sheet, copy its ID from the URL:
 *      docs.google.com/spreadsheets/d/SHEET_ID_HERE/edit
 *   3. Paste that ID as SHEET_ID below
 *   4. Make sure EMAIL_COLUMN matches the column letter for the email question
 *   5. Set trigger: Extensions → Apps Script → Triggers → Add Trigger
 *      Function: onFormSubmit | Event source: From form | Event type: On form submit
 */

const SHEET_ID     = "YOUR_GOOGLE_SHEET_ID_HERE";
const EMAIL_COLUMN = 2;    // column B (Forms puts Timestamp in A, responses start at B)
const MAX_SUBS     = 2000;

function onFormSubmit(e) {
  const sheet    = SpreadsheetApp.openById(SHEET_ID).getActiveSheet();
  const lastRow  = sheet.getLastRow();

  if (lastRow < 2) return; // nothing to validate yet

  // All emails in column B from row 2 onwards
  const emailRange = sheet.getRange(2, EMAIL_COLUMN, lastRow - 1, 1);
  const allEmails  = emailRange.getValues().flat().map(v => v.toString().trim().toLowerCase());
  const newEmail   = allEmails[allEmails.length - 1]; // just-submitted email

  // 1. Enforce 2000-subscriber cap
  const validCount = allEmails.filter(e => e.includes("@")).length;
  if (validCount > MAX_SUBS) {
    sheet.deleteRow(lastRow);
    Logger.log(`Cap reached (${MAX_SUBS}) — removed: ${newEmail}`);
    return;
  }

  // 2. Remove duplicate (keep first occurrence, delete this new row)
  const occurrences = allEmails.filter(e => e === newEmail).length;
  if (occurrences > 1) {
    sheet.deleteRow(lastRow);
    Logger.log(`Duplicate removed: ${newEmail}`);
    return;
  }

  Logger.log(`Subscriber added: ${newEmail} (${validCount}/${MAX_SUBS})`);
}
