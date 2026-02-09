/**
 * Math formatting utilities for terminal display
 *
 * Converts LaTeX-style notation to Unicode for better terminal rendering.
 */

/**
 * Common LaTeX to Unicode replacements
 */
const LATEX_UNICODE: [RegExp, string][] = [
  // Greek letters
  [/\\alpha/g, 'α'],
  [/\\beta/g, 'β'],
  [/\\gamma/g, 'γ'],
  [/\\delta/g, 'δ'],
  [/\\epsilon/g, 'ε'],
  [/\\zeta/g, 'ζ'],
  [/\\eta/g, 'η'],
  [/\\theta/g, 'θ'],
  [/\\iota/g, 'ι'],
  [/\\kappa/g, 'κ'],
  [/\\lambda/g, 'λ'],
  [/\\mu/g, 'μ'],
  [/\\nu/g, 'ν'],
  [/\\xi/g, 'ξ'],
  [/\\pi/g, 'π'],
  [/\\rho/g, 'ρ'],
  [/\\sigma/g, 'σ'],
  [/\\tau/g, 'τ'],
  [/\\upsilon/g, 'υ'],
  [/\\phi/g, 'φ'],
  [/\\chi/g, 'χ'],
  [/\\psi/g, 'ψ'],
  [/\\omega/g, 'ω'],
  [/\\Gamma/g, 'Γ'],
  [/\\Delta/g, 'Δ'],
  [/\\Theta/g, 'Θ'],
  [/\\Lambda/g, 'Λ'],
  [/\\Xi/g, 'Ξ'],
  [/\\Pi/g, 'Π'],
  [/\\Sigma/g, 'Σ'],
  [/\\Phi/g, 'Φ'],
  [/\\Psi/g, 'Ψ'],
  [/\\Omega/g, 'Ω'],

  // Logic and set theory (longer patterns before shorter to avoid prefix conflicts)
  [/\\forall/g, '∀'],
  [/\\nexists/g, '∄'],
  [/\\exists/g, '∃'],
  [/\\notin/g, '∉'],
  [/\\in(?![a-zA-Z])/g, '∈'],
  [/\\subseteq/g, '⊆'],
  [/\\subset/g, '⊂'],
  [/\\supseteq/g, '⊇'],
  [/\\supset/g, '⊃'],
  [/\\cup/g, '∪'],
  [/\\cap/g, '∩'],
  [/\\emptyset/g, '∅'],
  [/\\varnothing/g, '∅'],

  // Logical operators
  [/\\land/g, '∧'],
  [/\\lor/g, '∨'],
  [/\\lnot/g, '¬'],
  [/\\neg/g, '¬'],
  [/\\implies/g, '⟹'],
  [/\\Rightarrow/g, '⇒'],
  [/\\Leftarrow/g, '⇐'],
  [/\\Leftrightarrow/g, '⇔'],
  [/\\iff/g, '⟺'],
  [/\\rightarrow/g, '→'],
  [/\\leftarrow/g, '←'],
  [/\\leftrightarrow/g, '↔'],
  [/\\mapsto/g, '↦'],
  [/\\to(?![a-zA-Z])/g, '→'],

  // Relations (longer patterns before shorter to avoid prefix conflicts)
  [/\\leq/g, '≤'],
  [/\\geq/g, '≥'],
  [/\\neq/g, '≠'],
  [/\\equiv/g, '≡'],
  [/\\approx/g, '≈'],
  [/\\simeq/g, '≃'],
  [/\\sim(?![a-zA-Z])/g, '∼'],
  [/\\cong/g, '≅'],
  [/\\propto/g, '∝'],
  [/\\ll/g, '≪'],
  [/\\gg/g, '≫'],
  [/\\prec/g, '≺'],
  [/\\succ/g, '≻'],

  // Operators
  [/\\times/g, '×'],
  [/\\div/g, '÷'],
  [/\\cdot/g, '·'],
  [/\\pm/g, '±'],
  [/\\mp/g, '∓'],
  [/\\oplus/g, '⊕'],
  [/\\otimes/g, '⊗'],
  [/\\circ/g, '∘'],

  // Calculus and analysis
  [/\\sum/g, '∑'],
  [/\\prod/g, '∏'],
  [/\\int/g, '∫'],
  [/\\oint/g, '∮'],
  [/\\partial/g, '∂'],
  [/\\nabla/g, '∇'],
  [/\\infty/g, '∞'],
  [/\\lim/g, 'lim'],

  // Number sets
  [/\\mathbb\{N\}/g, 'ℕ'],
  [/\\mathbb\{Z\}/g, 'ℤ'],
  [/\\mathbb\{Q\}/g, 'ℚ'],
  [/\\mathbb\{R\}/g, 'ℝ'],
  [/\\mathbb\{C\}/g, 'ℂ'],

  // Geometry
  [/\\triangle/g, '△'],
  [/\\angle/g, '∠'],
  [/\\perp/g, '⊥'],
  [/\\parallel/g, '∥'],
  [/\\overline\{([^}]+)\}/g, '$1'],

  // Misc
  [/\\sqrt\{([^}]+)\}/g, '√($1)'],
  [/\\sqrt/g, '√'],
  [/\\therefore/g, '∴'],
  [/\\because/g, '∵'],
  [/\\ldots/g, '…'],
  [/\\cdots/g, '⋯'],
  [/\\vdots/g, '⋮'],
  [/\\ddots/g, '⋱'],
  [/\\qed/g, '∎'],
  [/\\quad/g, '  '],
  [/\\qquad/g, '    '],
  [/\\,/g, ' '],
  [/\\;/g, ' '],
  [/\\!/g, ''],

  // Line break
  [/\\\\/g, '\n'],
];

/**
 * Superscript and subscript numbers
 */
const SUPERSCRIPTS: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
  '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
  '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
  'n': 'ⁿ', 'i': 'ⁱ',
};

const SUBSCRIPTS: Record<string, string> = {
  '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
  '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
  '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
  'a': 'ₐ', 'e': 'ₑ', 'o': 'ₒ', 'x': 'ₓ',
  'i': 'ᵢ', 'j': 'ⱼ', 'k': 'ₖ', 'n': 'ₙ',
};

/**
 * Convert a string to superscript
 */
function toSuperscript(str: string): string {
  return str.split('').map(c => SUPERSCRIPTS[c] || c).join('');
}

/**
 * Convert a string to subscript
 */
function toSubscript(str: string): string {
  return str.split('').map(c => SUBSCRIPTS[c] || c).join('');
}

/**
 * Convert LaTeX-style math to Unicode for terminal display.
 *
 * @param latex  The LaTeX string to convert.
 * @param stripBraces  If true, remove leftover { } after processing.
 *                     Safe for delimited math; dangerous for full text.
 */
export function latexToUnicode(latex: string, stripBraces = false): string {
  let result = latex;

  // Structural commands first (need braces intact)
  // Handle fractions: \frac{a}{b} -> a/b
  result = result.replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)');

  // Handle \boxed{...} -> [ ... ]
  result = result.replace(/\\boxed\{([^}]+)\}/g, '[ $1 ]');

  // Handle text-mode commands
  result = result.replace(/\\text\{([^}]+)\}/g, '$1');
  result = result.replace(/\\mathrm\{([^}]+)\}/g, '$1');
  result = result.replace(/\\mathbf\{([^}]+)\}/g, '$1');
  result = result.replace(/\\mathit\{([^}]+)\}/g, '$1');
  result = result.replace(/\\mathbb\{([^}]+)\}/g, '$1');

  // Apply all symbol replacements (includes \sqrt{}, \overline{}, etc.)
  for (const [pattern, replacement] of LATEX_UNICODE) {
    result = result.replace(pattern, replacement);
  }

  // Handle superscripts: x^{12} -> x¹², x^2 -> x²
  result = result.replace(/\^\{([^}]+)\}/g, (_, content) => toSuperscript(content));
  result = result.replace(/\^(\d+)/g, (_, num) => toSuperscript(num));
  result = result.replace(/\^([a-zA-Z])/g, (_, char) => toSuperscript(char));

  // Handle subscripts: x_{12} -> x₁₂, x_1 -> x₁
  result = result.replace(/_\{([^}]+)\}/g, (_, content) => toSubscript(content));
  result = result.replace(/_(\d+)/g, (_, num) => toSubscript(num));
  result = result.replace(/_([a-zA-Z])/g, (_, char) => toSubscript(char));

  // Clean up remaining \command sequences
  result = result.replace(/\\([a-zA-Z]+)/g, '$1');

  // Optionally strip leftover braces (safe inside delimited math only)
  if (stripBraces) {
    result = result.replace(/[{}]/g, '');
  }

  return result;
}

/**
 * Format a mathematical expression for display
 */
export function formatMathExpression(expr: string): string {
  // Try to detect if it's LaTeX
  if (expr.includes('\\') || expr.includes('^{') || expr.includes('_{')) {
    return latexToUnicode(expr);
  }

  // Simple transformations for plain expressions
  let result = expr;

  // Simple power notation: n^2 -> n²
  result = result.replace(/\^(\d)/g, (_, d) => toSuperscript(d));

  // Sum notation: sum_{i=1}^n -> Σᵢ₌₁ⁿ (simplified)
  result = result.replace(/sum_?\{?([^}]+)\}?/gi, (_, range) => `Σ(${range})`);

  return result;
}

/**
 * Format a proof step with proper math rendering
 */
export function formatProofStep(step: string): string {
  // Process inline math between $ signs
  return step.replace(/\$([^$]+)\$/g, (_, math) => latexToUnicode(math));
}
