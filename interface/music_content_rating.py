import pandas as pd
import argparse
from pathlib import Path
import numpy as np


class MusicContentRatingSystem:
    def __init__(self):
        # Primary rating thresholds
        self.rating_thresholds = {
            'M-E': {'max': 0.05},
            'M-P': {'min': 0.05, 'max': 0.40},
            'M-T': {'min': 0.40, 'max': 0.70},
            'M-R': {'min': 0.70, 'max': 0.95},
            'M-AO': {'min': 0.95}
        }
        #AO conditions
        self.ao_special_conditions = {
            'single_extreme': 0.95,  # Any score > 0.95
            'multiple_high': 0.85    # Two or more scores > 0.85
        }

    def calculate_rating(self, violence_score, sexual_score, language_score, substance_score):
        """Calculate overall rating and descriptors."""
        scores = {
            'violence': float(violence_score),
            'sexual': float(sexual_score),
            'language': float(language_score),
            'substance': float(substance_score)
        }

        # Clip scores to [0, 1]
        for k in scores:
            scores[k] = float(np.clip(scores[k], 0.0, 1.0))

        max_score = max(scores.values())

        # AO conditions first
        if self._check_ao_conditions(scores):
            rating = 'M-AO'
        else:
            rating = self._determine_rating_by_max(max_score)

        descriptors = self._generate_descriptors(scores)

        return {
            'rating': rating,
            'scores': scores,
            'descriptors': descriptors,
            'max_score': max_score,
            'details': self._generate_details(scores, rating)
        }

    def _determine_rating_by_max(self, max_score: float) -> str:
        if max_score <= self.rating_thresholds['M-E']['max']:
            return 'M-E'
        if self.rating_thresholds['M-P']['min'] < max_score <= self.rating_thresholds['M-P']['max']:
            return 'M-P'
        if self.rating_thresholds['M-T']['min'] < max_score <= self.rating_thresholds['M-T']['max']:
            return 'M-T'
        if self.rating_thresholds['M-R']['min'] < max_score <= self.rating_thresholds['M-R']['max']:
            return 'M-R'
        return 'M-AO'

    def _check_ao_conditions(self, scores):
        if any(v > self.ao_special_conditions['single_extreme'] for v in scores.values()):
            return True
        if sum(v > self.ao_special_conditions['multiple_high'] for v in scores.values()) >= 2:
            return True
        return False

    def _generate_descriptors(self, scores):
        """Generate content descriptors based on thresholds."""
        descriptors = []

        # Language
        lang = scores['language']
        if lang > 0.70:
            descriptors.append('Explicit Language')
        elif lang > 0.40:
            descriptors.append('Strong Language')
        elif 0.05 < lang <= 0.40:
            descriptors.append('Mild Language')

        # Violence
        vio = scores['violence']
        if vio > 0.95:
            descriptors.append('Graphic Violence')
        elif vio > 0.70:
            descriptors.append('Intense Violence')
        elif vio > 0.40:
            descriptors.append('Violence')

        # Sexual
        sex = scores['sexual']
        if sex > 0.95:
            descriptors.append('Graphic Sexual Content')
        elif sex > 0.70:
            descriptors.append('Sexual Content')
        elif sex > 0.40:
            descriptors.append('Suggestive Themes')

        # Substance
        sub = scores['substance']
        if sub > 0.95:
            descriptors.append('Glorified Drug Use')
        elif sub > 0.70:
            descriptors.append('Drug Abuse')
        elif sub > 0.40:
            descriptors.append('Drug Reference')

        return descriptors

    def _generate_details(self, scores, rating):
        return {
            'rating_explanation': self._get_rating_explanation(rating),
            'highest_concern': max(scores, key=scores.get),
            'all_scores': scores,
            'recommendation': self._get_recommendation(rating)
        }

    def _get_rating_explanation(self, rating):
        explanations = {
            'M-E': 'Content generally suitable for all ages',
            'M-P': 'Some material may not be suitable for children',
            'M-T': 'Content generally suitable for ages 13 and up',
            'M-R': 'Under 17 requires accompanying parent or adult guardian',
            'M-AO': 'Content suitable only for adults ages 18 and up'
        }
        return explanations.get(rating, '')

    def _get_recommendation(self, rating):
        recommendations = {
            'M-E': 'Safe for family listening',
            'M-P': 'Parents should review before allowing young children to listen',
            'M-T': 'Appropriate for ages 13+ teenagers',
            'M-R': 'Parental discretion strongly advised',
            'M-AO': 'Adults only - not suitable for minors'
        }
        return recommendations.get(rating, '')


def _clean_scores(df: pd.DataFrame, cols):
    """Convert to float, fill missing with 0, clip to [0,1]."""
    cleaned = df.copy()
    for col in cols:
        original = cleaned[col]
        coerced = pd.to_numeric(original, errors='coerce')
        non_numeric = coerced.isna() & original.notna()
        missing_before = original.isna().sum()

        if non_numeric.any():
            print(f"[Warning] Column '{col}': {non_numeric.sum()} non-numeric values coerced to NaN.")
        if missing_before > 0:
            print(f"[Warning] Column '{col}': {missing_before} missing values detected.")

        coerced = coerced.fillna(0.0)
        out_low = (coerced < 0).sum()
        out_high = (coerced > 1).sum()
        if out_low or out_high:
            print(f"[Warning] Column '{col}': {out_low + out_high} values out of [0,1] clipped.")

        cleaned[col] = np.clip(coerced, 0.0, 1.0)
    return cleaned


def _format_descriptors_for_output(result):
    """
    Replace empty descriptors with human-friendly text:
    - M-E -> 'Everyone'
    - M-P -> 'Parental Guidance Suggested'
    - otherwise 'None'
    """
    if result['descriptors']:
        return '; '.join(result['descriptors'])
    if result['rating'] == 'M-E':
        return 'Everyone'
    if result['rating'] == 'M-P':
        return 'Parental Guidance Suggested'
    return 'None'


def rate_single_song(violence, sexual, language, substance):
    """Rate a single song from four scores."""
    rater = MusicContentRatingSystem()
    result = rater.calculate_rating(violence, sexual, language, substance)

    descriptors_text = _format_descriptors_for_output(result)
    final_rating = (
        f"{result['rating']}: {descriptors_text}"
        if (result['rating'] != 'M-E' and result['descriptors'])
        else result['rating']
    )

    print("\nMusic Content Rating Result:")
    print("-" * 50)
    print("Scores:")
    print(f"Violence: {result['scores']['violence']:.3f}")
    print(f"Sexual: {result['scores']['sexual']:.3f}")
    print(f"Language: {result['scores']['language']:.3f}")
    print(f"Substance: {result['scores']['substance']:.3f}")
    print(f"\nMaximum Score: {result['max_score']:.3f}")
    print(f"Primary Rating: {result['rating']}")
    print(f"Content Descriptors: {descriptors_text}")
    print(f"Final Rating: {final_rating}")
    print(f"Recommendation: {result['details']['recommendation']}")
    return result


def process_csv_file(input_file, output_file=None,
                     violence_col='violence_score',
                     sexual_col='sexual_score',
                     language_col='language_score',
                     substance_col='substance_score'):
    """Process a CSV file: clean, rate, export."""
    try:
        df = pd.read_csv(input_file)
        if 'song_index' not in df.columns:
            df.insert(0, 'song_index', df.index)
        print(f"Loaded file: {input_file}")
        print(f"Total {len(df)} rows\n")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    required = [violence_col, sexual_col, language_col, substance_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Clean scores
    df = _clean_scores(df, required)
    rater = MusicContentRatingSystem()

    results_df = df.copy()

    # Init result columns
    results_df['MCR_rating'] = ''
    results_df['MCR_descriptors'] = ''
    results_df['MCR_max_score'] = 0.0
    results_df['MCR_highest_concern'] = ''
    results_df['MCR_recommendation'] = ''
    results_df['MCR_final_rating'] = ''

    # Rate each row
    for idx, row in df.iterrows():
        v, s, l, d = row[violence_col], row[sexual_col], row[language_col], row[substance_col]
        result = rater.calculate_rating(v, s, l, d)

        descriptors_text = _format_descriptors_for_output(result)
        final_rating = (
            f"{result['rating']}: {descriptors_text}"
            if (result['rating'] != 'M-E' and result['descriptors'])
            else result['rating']
        )

        results_df.at[idx, 'MCR_rating'] = result['rating']
        results_df.at[idx, 'MCR_descriptors'] = descriptors_text
        results_df.at[idx, 'MCR_max_score'] = result['max_score']
        results_df.at[idx, 'MCR_highest_concern'] = result['details']['highest_concern']
        results_df.at[idx, 'MCR_recommendation'] = result['details']['recommendation']
        results_df.at[idx, 'MCR_final_rating'] = final_rating

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(df)}")


    # Column order for export
    keep_cols = [
        'song_index', 'song', 'text',
        'sexual_score', 'violence_score', 'substance_score', 'language_score',
        'sexual', 'violence', 'substance', 'language',
        'explicit_edit',
        'MCR_rating', 'MCR_descriptors', 'MCR_max_score',
        'MCR_highest_concern', 'MCR_recommendation', 'MCR_final_rating'
    ]

    for c in keep_cols:
        if c not in results_df.columns:
            results_df[c] = ''

    # keep only these columns
    results_df = results_df[keep_cols]

    # Save
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        input_path = Path(input_file)
        output_path = input_path.parent / f"{input_path.stem}_MCR_rated.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    # Stats
    print("\nRating Statistics:")
    rating_counts = results_df['MCR_rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        pct = (count / len(results_df)) * 100
        print(f"  {rating}: {count} ({pct:.1f}%)")

    print("\nMost Common Content Descriptors:")
    all_desc = []
    for desc in results_df['MCR_descriptors']:
        if desc != 'None':
            all_desc.extend(desc.split('; '))
    if all_desc:
        desc_counts = pd.Series(all_desc).value_counts().head(10)
        for k, v in desc_counts.items():
            print(f"  {k}: {v}")

    return results_df

