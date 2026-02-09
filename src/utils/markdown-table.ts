/**
 * Markdown rendering for terminal.
 *
 * Simple terminal markdown rendering without external syntax highlighting.
 * Handles: headers, bold, italic, code blocks, lists, horizontal rules.
 * Plus custom LaTeX to Unicode conversion.
 */

import chalk from 'chalk';
import { latexToUnicode } from './math-format.js';

// Box-drawing characters
const BOX = {
  topLeft: '┌',
  topRight: '┐',
  bottomLeft: '└',
  bottomRight: '┘',
  horizontal: '─',
  vertical: '│',
  topT: '┬',
  bottomT: '┴',
  leftT: '├',
  rightT: '┤',
  cross: '┼',
};

/**
 * Check if a string looks like a number (for right-alignment).
 */
function isNumeric(value: string): boolean {
  const trimmed = value.trim();
  // Match numbers with optional $, %, B/M/K suffixes
  return /^[$]?[-+]?[\d,]+\.?\d*[%BMK]?$/.test(trimmed);
}

/**
 * Parse a markdown table into headers and rows.
 */
export function parseMarkdownTable(tableText: string): { headers: string[]; rows: string[][] } | null {
  const lines = tableText.trim().split('\n').map(line => line.trim());
  
  if (lines.length < 2) return null;
  
  // Parse header line
  const headerLine = lines[0];
  if (!headerLine.includes('|')) return null;
  
  const headers = headerLine
    .split('|')
    .map(cell => cell.trim())
    .filter((_, i, arr) => i > 0 && i < arr.length - 1 || arr.length === 1);
  
  // Handle edge case where there's no leading/trailing pipe
  if (headers.length === 0) {
    const rawHeaders = headerLine.split('|').map(cell => cell.trim());
    if (rawHeaders.length > 0) {
      headers.push(...rawHeaders);
    }
  }
  
  if (headers.length === 0) return null;
  
  // Check for separator line (---|---|---)
  const separatorLine = lines[1];
  if (!separatorLine || !/^[\s|:-]+$/.test(separatorLine)) return null;
  
  // Parse data rows
  const rows: string[][] = [];
  for (let i = 2; i < lines.length; i++) {
    const line = lines[i];
    if (!line.includes('|')) continue;
    
    const cells = line
      .split('|')
      .map(cell => cell.trim());
    
    // Remove empty first/last cells from pipes at start/end
    if (cells[0] === '') cells.shift();
    if (cells[cells.length - 1] === '') cells.pop();
    
    if (cells.length > 0) {
      rows.push(cells);
    }
  }
  
  return { headers, rows };
}

/**
 * Render a parsed table as a Unicode box-drawing table.
 */
export function renderBoxTable(headers: string[], rows: string[][]): string {
  // Calculate column widths
  const colWidths: number[] = headers.map(h => h.length);
  
  for (const row of rows) {
    for (let i = 0; i < row.length; i++) {
      if (i < colWidths.length) {
        colWidths[i] = Math.max(colWidths[i], row[i].length);
      }
    }
  }
  
  // Determine alignment for each column (right for numeric, left for text)
  const alignRight: boolean[] = headers.map((_, colIndex) => {
    // Check if most values in this column are numeric
    let numericCount = 0;
    for (const row of rows) {
      if (row[colIndex] && isNumeric(row[colIndex])) {
        numericCount++;
      }
    }
    return numericCount > rows.length / 2;
  });
  
  // Helper to pad a cell
  const padCell = (value: string, width: number, rightAlign: boolean): string => {
    if (rightAlign) {
      return value.padStart(width);
    }
    return value.padEnd(width);
  };
  
  // Build the table
  const lines: string[] = [];
  
  // Top border
  const topBorder = BOX.topLeft + 
    colWidths.map(w => BOX.horizontal.repeat(w + 2)).join(BOX.topT) + 
    BOX.topRight;
  lines.push(topBorder);
  
  // Header row
  const headerRow = BOX.vertical + 
    headers.map((h, i) => ` ${padCell(h, colWidths[i], false)} `).join(BOX.vertical) + 
    BOX.vertical;
  lines.push(headerRow);
  
  // Header separator
  const headerSep = BOX.leftT + 
    colWidths.map(w => BOX.horizontal.repeat(w + 2)).join(BOX.cross) + 
    BOX.rightT;
  lines.push(headerSep);
  
  // Data rows
  for (const row of rows) {
    const dataRow = BOX.vertical + 
      colWidths.map((w, i) => {
        const value = row[i] || '';
        return ` ${padCell(value, w, alignRight[i])} `;
      }).join(BOX.vertical) + 
      BOX.vertical;
    lines.push(dataRow);
  }
  
  // Bottom border
  const bottomBorder = BOX.bottomLeft + 
    colWidths.map(w => BOX.horizontal.repeat(w + 2)).join(BOX.bottomT) + 
    BOX.bottomRight;
  lines.push(bottomBorder);
  
  return lines.join('\n');
}

/**
 * Find and transform all markdown tables in content to box-drawing tables.
 */
export function transformMarkdownTables(content: string): string {
  // Normalize line endings: convert \r\n to \n, then trim trailing whitespace from each line
  const normalized = content
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map(line => line.trimEnd())
    .join('\n');
  
  // Regex to match markdown tables:
  // - Starts with a line containing pipes
  // - Followed by a separator line (---|---|---)
  // - Followed by zero or more data rows with pipes
  // IMPORTANT: Use [ \t] instead of \s in separator to avoid matching newlines
  const tableRegex = /^(\|[^\n]+\|\n\|[-:| \t]+\|(?:\n\|[^\n]+\|)*)/gm;
  
  // Also match tables without leading/trailing pipes on each line
  const tableRegex2 = /^([^\n|]*\|[^\n]+\n[-:| \t]+(?:\n[^\n|]*\|[^\n]+)*)/gm;
  
  let result = normalized;
  
  // Process tables with pipes at start/end
  result = result.replace(tableRegex, (match) => {
    const parsed = parseMarkdownTable(match);
    if (parsed && parsed.headers.length > 0 && parsed.rows.length > 0) {
      return renderBoxTable(parsed.headers, parsed.rows);
    }
    return match;
  });
  
  // Process tables that might not have leading pipes
  result = result.replace(tableRegex2, (match) => {
    // Skip if already transformed (contains box-drawing chars)
    if (match.includes(BOX.topLeft)) return match;
    
    const parsed = parseMarkdownTable(match);
    if (parsed && parsed.headers.length > 0 && parsed.rows.length > 0) {
      return renderBoxTable(parsed.headers, parsed.rows);
    }
    return match;
  });
  
  return result;
}

/**
 * Transform markdown bold (**text**) to ANSI bold.
 */
export function transformBold(content: string): string {
  return content.replace(/\*\*([^*]+)\*\*/g, (_, text) => chalk.bold(text));
}

/**
 * Apply all pre-render formatting to response content.
 * - Converts markdown tables to unicode box-drawing tables
 * - Converts **bold** to ANSI bold
 */
/**
 * Known LaTeX commands → Unicode for bare (undelimited) usage.
 * Only safe, unambiguous symbol substitutions — no structural
 * transforms (fractions, superscripts, braces, etc.).
 */
const BARE_LATEX: [RegExp, string][] = [
  // Geometry
  [/\\triangle(?![a-zA-Z])/g, '△'],
  [/\\angle(?![a-zA-Z])/g, '∠'],
  [/\\perp(?![a-zA-Z])/g, '⊥'],
  [/\\parallel(?![a-zA-Z])/g, '∥'],
  // Relations
  [/\\cong(?![a-zA-Z])/g, '≅'],
  [/\\sim(?![a-zA-Z])/g, '∼'],
  [/\\simeq(?![a-zA-Z])/g, '≃'],
  [/\\equiv(?![a-zA-Z])/g, '≡'],
  [/\\approx(?![a-zA-Z])/g, '≈'],
  [/\\neq(?![a-zA-Z])/g, '≠'],
  [/\\leq(?![a-zA-Z])/g, '≤'],
  [/\\geq(?![a-zA-Z])/g, '≥'],
  [/\\ll(?![a-zA-Z])/g, '≪'],
  [/\\gg(?![a-zA-Z])/g, '≫'],
  [/\\propto(?![a-zA-Z])/g, '∝'],
  // Operators
  [/\\cdot(?![a-zA-Z])/g, '·'],
  [/\\times(?![a-zA-Z])/g, '×'],
  [/\\div(?![a-zA-Z])/g, '÷'],
  [/\\pm(?![a-zA-Z])/g, '±'],
  // Calculus
  [/\\infty(?![a-zA-Z])/g, '∞'],
  [/\\partial(?![a-zA-Z])/g, '∂'],
  [/\\nabla(?![a-zA-Z])/g, '∇'],
  [/\\sum(?![a-zA-Z])/g, '∑'],
  [/\\prod(?![a-zA-Z])/g, '∏'],
  [/\\int(?![a-zA-Z])/g, '∫'],
  // Greek (common)
  [/\\pi(?![a-zA-Z])/g, 'π'],
  [/\\sigma(?![a-zA-Z])/g, 'σ'],
  [/\\Sigma(?![a-zA-Z])/g, 'Σ'],
  [/\\alpha(?![a-zA-Z])/g, 'α'],
  [/\\beta(?![a-zA-Z])/g, 'β'],
  [/\\gamma(?![a-zA-Z])/g, 'γ'],
  [/\\delta(?![a-zA-Z])/g, 'δ'],
  [/\\Delta(?![a-zA-Z])/g, 'Δ'],
  [/\\epsilon(?![a-zA-Z])/g, 'ε'],
  [/\\theta(?![a-zA-Z])/g, 'θ'],
  [/\\lambda(?![a-zA-Z])/g, 'λ'],
  [/\\mu(?![a-zA-Z])/g, 'μ'],
  [/\\phi(?![a-zA-Z])/g, 'φ'],
  [/\\omega(?![a-zA-Z])/g, 'ω'],
  [/\\Omega(?![a-zA-Z])/g, 'Ω'],
  // Logic
  [/\\forall(?![a-zA-Z])/g, '∀'],
  [/\\exists(?![a-zA-Z])/g, '∃'],
  [/\\in(?![a-zA-Z])/g, '∈'],
  [/\\implies(?![a-zA-Z])/g, '⟹'],
  [/\\iff(?![a-zA-Z])/g, '⟺'],
  [/\\to(?![a-zA-Z])/g, '→'],
  // Misc
  [/\\ldots(?![a-zA-Z])/g, '…'],
  [/\\cdots(?![a-zA-Z])/g, '⋯'],
  [/\\therefore(?![a-zA-Z])/g, '∴'],
  [/\\qed(?![a-zA-Z])/g, '∎'],
  // Structural (safe without braces context)
  [/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)'],
  [/\\boxed\{([^}]+)\}/g, '[ $1 ]'],
  [/\\text\{([^}]+)\}/g, '$1'],
  [/\\sqrt\{([^}]+)\}/g, '√($1)'],
  [/\\overline\{([^}]+)\}/g, '$1'],
];

/**
 * Convert LaTeX math to Unicode.
 *
 * Handles:
 *  - Display math: $$...$$ and \[...\]
 *  - Inline math: $...$
 *  - Bare LaTeX commands outside delimiters (e.g. \triangle)
 */
function transformMath(text: string): string {
  // Strip display math delimiters: $$...$$ → content, \[...\] → content
  let result = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math: string) =>
    latexToUnicode(math.trim(), true)
  );
  result = result.replace(/\\\[([\s\S]*?)\\\]/g, (_, math: string) =>
    latexToUnicode(math.trim(), true)
  );

  // Process inline math $...$
  result = result.replace(/\$([^$]+)\$/g, (_, math: string) =>
    latexToUnicode(math, true)
  );

  // Handle bare LaTeX commands outside any delimiters.
  // Only apply safe symbol substitutions — no structural transforms
  // that could corrupt non-math text.
  for (const [pattern, replacement] of BARE_LATEX) {
    result = result.replace(pattern, replacement);
  }

  return result;
}

/**
 * Simple terminal markdown renderer without external highlighting libraries.
 */
function renderMarkdownForTerminal(text: string): string {
  let result = text;

  // Headers: ## Text -> bold cyan text (remove ##)
  result = result.replace(/^#{1,6}\s+(.+)$/gm, (_, content) =>
    chalk.bold.cyan(content)
  );

  // Code blocks: ```lang ... ``` -> indented cyan
  result = result.replace(/```[\w]*\n([\s\S]*?)```/g, (_, code: string) => {
    const indented = code.trim().split('\n').map(line => '    ' + line).join('\n');
    return '\n' + chalk.cyan(indented) + '\n';
  });

  // Inline code: `code` -> cyan
  result = result.replace(/`([^`]+)`/g, (_, code) => chalk.cyan(code));

  // Horizontal rules: --- -> line
  result = result.replace(/^---+$/gm, chalk.gray('─'.repeat(50)));

  // Lists: - item or * item -> bullet
  result = result.replace(/^[\s]*[-*]\s+(.+)$/gm, (_, content) =>
    '  • ' + content
  );

  // Numbered lists: 1. item -> number with dot
  result = result.replace(/^[\s]*(\d+)\.\s+(.+)$/gm, (_, num, content) =>
    `  ${num}. ${content}`
  );

  // Blockquotes: > text -> gray italic
  result = result.replace(/^>\s+(.+)$/gm, (_, content) =>
    chalk.gray.italic('│ ' + content)
  );

  // Links: [text](url) -> text (underlined blue url)
  result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, linkText, url) =>
    linkText + ' ' + chalk.blue.underline('(' + url + ')')
  );

  return result;
}

export function formatResponse(content: string): string {
  let result = content;

  // Convert all LaTeX (display $$, inline $, bare commands) to Unicode
  result = transformMath(result);

  // Transform markdown tables to box-drawing tables
  result = transformMarkdownTables(result);

  // Apply simple terminal markdown rendering
  result = renderMarkdownForTerminal(result);

  // Transform **bold** to ANSI bold (after markdown rendering for nested cases)
  result = transformBold(result);

  return result;
}
