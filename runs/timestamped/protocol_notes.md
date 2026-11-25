## Evaluation correction

After an initial evaluation, a deeper check found five exact duplicate input
pairs linking house11 and house19: 234/389, 235/390, 236/391, 237/392 and 238/393.
The required electrical measurements and time/sample columns match across each
pair. Both houses are in development; none of the 215 test recordings is part
of these duplicate groups.

Development cross-validation was corrected to keep the connected houses in
the same fold. The same 12 candidate settings were compared again, and the
corrected validation scores selected `extra_trees_half`. Test houses and IDs
were unchanged. The final result is a re-evaluation after this data-integrity
fix. Test scores were not used to select the corrected model, but the test set
had previously been inspected. The first result must not be treated as an
independent second test set or combined with the final result.
