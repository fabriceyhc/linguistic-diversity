"""A fixed corpus of mutually unrelated sentences, used to calibrate the similarity floor.

Sentence encoders do not send unrelated text to orthogonal vectors. They send it to
some positive baseline, and because every document is slightly similar to every
*other* document that baseline accumulates: the largest diversity any abundance
distribution can reach is ``n / (1 + (n-1)z)``, tending to ``1/z``. A floor of 0.42
caps a corpus at roughly 2.4 effective species however large it is, which is what
stops the score meaning "effective number of things said".

Measuring that baseline needs text that is unrelated *by construction*. These
sentences span topic, register, length, tense and syntactic frame, with no shared
subject matter between any two -- so their mean pairwise similarity estimates the
encoder's floor rather than any property of a corpus under study.

The corpus is fixed and shipped so the estimate is a constant of the encoder. It
must never be replaced by the user's own corpus: a floor estimated from the text
being measured makes each pair's similarity depend on unrelated documents, which
is exactly the defect that cost ``mean_adj`` its replication invariance.
"""

from __future__ import annotations

# 60 sentences, no two on the same subject. Lengths run from 4 to 30 words, since
# an encoder's baseline can drift with document length.
REFERENCE_CORPUS: tuple[str, ...] = (
    "The kettle boiled dry while nobody was watching.",
    "Sedimentary layers record the slow retreat of an inland sea.",
    "She refused the promotion for reasons she never explained.",
    "Copper prices fell sharply after the strike was announced.",
    "A barn owl hunts by sound alone in total darkness.",
    "The compiler rejects any function that shadows a builtin name.",
    "Rain had been falling on the tin roof since Tuesday.",
    "Medieval scribes ruled their parchment with a blunt stylus before writing.",
    "He plays the trombone badly but with tremendous enthusiasm.",
    "The referendum passed by fewer than four thousand votes.",
    "Yeast converts sugar into alcohol and carbon dioxide.",
    "Nobody has repaired the bridge since the flood.",
    "Her thesis concerned the phonology of a nearly extinct dialect.",
    "The cat knocked the glass off the table deliberately.",
    "Antarctic ice cores preserve air from eight hundred thousand years ago.",
    "They serve breakfast until eleven on weekends.",
    "The algorithm terminates only when the residual falls below tolerance.",
    "A wildfire closed the highway for three days in August.",
    "His grandmother knitted every jumper he owned until he turned twelve.",
    "Neutron stars can spin hundreds of times per second.",
    "The lease expires at the end of March.",
    "Cheddar was originally aged in caves at a constant temperature.",
    "She won the constituency on her fourth attempt.",
    "Static electricity ruined an entire batch of memory chips.",
    "The novel was rejected by nineteen publishers before anyone bought it.",
    "Migrating geese navigate partly by the earth's magnetic field.",
    "He forgot his umbrella and arrived completely soaked.",
    "Tax revenue exceeded the treasury forecast by a small margin.",
    "The kiln must cool slowly or the glaze will craze.",
    "Sharks have been swimming in these waters since before trees existed.",
    "Her landlord raised the rent without giving proper notice.",
    "The orchestra tuned to an oboe playing a single sustained note.",
    "Permafrost thaw is releasing methane across the Siberian plain.",
    "They painted the shed an alarming shade of green.",
    "A single misplaced semicolon delayed the launch by a week.",
    "The tide goes out far enough to walk to the island.",
    "Roman concrete grows stronger when seawater seeps into its cracks.",
    "She quit smoking the day her daughter was born.",
    "The auction house withdrew the painting over questions of provenance.",
    "Honeybees communicate the direction of food by dancing.",
    "The train was cancelled and no replacement bus appeared.",
    "Cursive handwriting is no longer taught in most primary schools.",
    "He built the whole cabin from timber felled on the property.",
    "Sunspot activity follows a cycle of roughly eleven years.",
    "The recipe calls for far more butter than seems reasonable.",
    "Archaeologists found a hoard of coins beneath a car park.",
    "Her flight was diverted to an airport four hundred miles away.",
    "Certain fungi can conduct electrical signals between distant roots.",
    "The committee met for six hours and decided nothing at all.",
    "Sea otters wrap themselves in kelp so they do not drift while sleeping.",
    "He learned Portuguese from watching football commentary.",
    "The clock in the hall has been ten minutes fast for years.",
    "Glass is not a slow-moving liquid, whatever the tour guide says.",
    "They demolished the factory to build apartments nobody can afford.",
    "A sourdough starter can outlive the baker who created it.",
    "The signal was traced to a microwave oven in the staff kitchen.",
    "Volcanic ash grounded flights across northern Europe that spring.",
    "She reads the last page of a novel before starting chapter one.",
    "Lightning strikes the same tall structures repeatedly and always has.",
    "The border was drawn by a civil servant who had never visited the region.",
)
