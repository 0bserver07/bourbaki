/**
 * Lean error patterns and their plain English translations
 */
interface ErrorPattern {
  pattern: RegExp;
  translate: (match: RegExpMatchArray) => string;
  suggestion?: string;
}

const ERROR_PATTERNS: ErrorPattern[] = [
  // Type mismatch
  {
    pattern: /type mismatch\s+(.+?)\s+has type\s+(.+?)\s+but is expected to have type\s+(.+)/s,
    translate: (m) => `Type mismatch: Got "${m[2]?.trim()}" but expected "${m[3]?.trim()}"`,
    suggestion: 'Check that you are using the right types. You may need to add explicit type annotations or conversions.',
  },

  // Unknown identifier
  {
    pattern: /unknown identifier '(.+?)'/,
    translate: (m) => `Unknown identifier: "${m[1]}" is not defined`,
    suggestion: 'Check spelling, or you may need to import a library. Try adding "import Mathlib" at the top.',
  },

  // Unknown tactic
  {
    pattern: /unknown tactic '(.+?)'/,
    translate: (m) => `Unknown tactic: "${m[1]}" is not a valid Lean tactic`,
    suggestion: 'Check the tactic name spelling. Common tactics: simp, ring, omega, linarith, rfl, exact, apply.',
  },

  // Unsolved goals
  {
    pattern: /unsolved goals/,
    translate: () => 'Proof incomplete: There are still goals left to prove',
    suggestion: 'The proof is not finished. You need to prove the remaining goals shown above.',
  },

  // Function has arguments but doesn't have body
  {
    pattern: /function has arguments? but doesn't have a body/,
    translate: () => 'Missing proof body: The theorem/lemma needs a proof',
    suggestion: 'Add ":= by" followed by tactics, or ":=" followed by a term-mode proof.',
  },

  // Expected token
  {
    pattern: /expected '(.+?)'/,
    translate: (m) => `Syntax error: Expected "${m[1]}"`,
    suggestion: 'Check your syntax. You may be missing a keyword, parenthesis, or operator.',
  },

  // Failed to synthesize
  {
    pattern: /failed to synthesize\s+(.+)/,
    translate: (m) => `Cannot find instance: Lean cannot automatically derive "${m[1]?.trim()}"`,
    suggestion: 'You may need to add explicit type class instances or use a different approach.',
  },

  // Invalid match
  {
    pattern: /invalid match-expression/,
    translate: () => 'Invalid pattern match: The cases do not cover all possibilities',
    suggestion: 'Make sure you handle all cases in your match expression.',
  },

  // Application type mismatch
  {
    pattern: /application type mismatch\s+(.+?)\s+argument\s+(.+?)\s+has type/s,
    translate: (m) => `Wrong argument type for "${m[1]?.trim()}"`,
    suggestion: 'Check that the arguments you are passing have the correct types.',
  },

  // Omega failed
  {
    pattern: /omega failed/,
    translate: () => 'Omega tactic failed: The statement is not linear arithmetic or is false',
    suggestion: 'Try using linarith, nlinarith (for nonlinear), or break down the proof manually.',
  },

  // Ring failed
  {
    pattern: /ring failed/,
    translate: () => 'Ring tactic failed: The equality is not purely algebraic',
    suggestion: 'The equation may need simplification first. Try "simp" before "ring", or add field_simp for fractions.',
  },

  // Simp made no progress
  {
    pattern: /simp made no progress/,
    translate: () => 'Simp had no effect: The simplifier could not simplify this expression',
    suggestion: 'Try simp with specific lemmas: "simp [lemma1, lemma2]", or try a different approach.',
  },

  // Not a function
  {
    pattern: /function expected/,
    translate: () => 'Not a function: Trying to apply something that is not a function',
    suggestion: 'Check that you are calling an actual function and that arguments are correct.',
  },

  // Induction error
  {
    pattern: /induction tactic failed/,
    translate: () => 'Induction failed: Cannot apply induction here',
    suggestion: 'Make sure the variable is the right type (usually Nat or a recursive type). Try "induction n with k ih".',
  },
];

/**
 * Translate a Lean error message to plain English
 */
export function translateLeanError(error: string): {
  original: string;
  translated: string;
  suggestion?: string;
} {
  for (const { pattern, translate, suggestion } of ERROR_PATTERNS) {
    const match = error.match(pattern);
    if (match) {
      return {
        original: error,
        translated: translate(match),
        suggestion,
      };
    }
  }

  // No pattern matched - return original with generic advice
  return {
    original: error,
    translated: error,
    suggestion: 'Check the Lean 4 documentation or try simplifying your proof.',
  };
}

/**
 * Format a Lean error for display
 */
export function formatLeanError(error: string): string {
  const { translated, suggestion } = translateLeanError(error);

  let output = `**Error:** ${translated}`;
  if (suggestion) {
    output += `\n**Suggestion:** ${suggestion}`;
  }

  return output;
}

/**
 * Extract and translate all errors from Lean output
 */
export function translateLeanOutput(output: string): string[] {
  const errors: string[] = [];
  const errorRegex = /error:\s*(.+?)(?=\n\n|\nerror:|\n[a-zA-Z]|$)/gs;

  let match;
  while ((match = errorRegex.exec(output)) !== null) {
    const { translated, suggestion } = translateLeanError(match[1].trim());
    let formatted = `• ${translated}`;
    if (suggestion) {
      formatted += `\n  → ${suggestion}`;
    }
    errors.push(formatted);
  }

  return errors;
}
