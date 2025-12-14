"""
String Matching Module for Smart Crowd Router.
Implements fuzzy string matching to find the closest places and streets.
Uses Levenshtein distance and other similarity metrics.
"""


def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    This is the minimum number of single-character edits (insertions, deletions, substitutions)
    needed to transform one string into another.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def similarity_ratio(s1, s2):
    """
    Calculate similarity ratio between two strings (0.0 to 1.0).
    Based on Levenshtein distance, normalized by the length of the longer string.
    """
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()
    
    if s1_lower == s2_lower:
        return 1.0
    
    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1_lower, s2_lower)
    return 1.0 - (distance / max_len)


def contains_bonus(query, candidate):
    """
    Give bonus points if the query is contained in the candidate or vice versa.
    """
    query_lower = query.lower().strip()
    candidate_lower = candidate.lower().strip()
    
    if query_lower in candidate_lower or candidate_lower in query_lower:
        return 0.15
    return 0.0


def word_match_bonus(query, candidate):
    """
    Give bonus points for matching words.
    """
    query_words = set(query.lower().strip().split())
    candidate_words = set(candidate.lower().strip().split())
    
    if not query_words or not candidate_words:
        return 0.0
    
    common_words = query_words & candidate_words
    filler_words = {'de', 'del', 'la', 'el', 'les', 'los', 'las', 'en', "d'", "l'", 'i', 'a', 'the', 'of'}
    meaningful_common = common_words - filler_words
    
    if meaningful_common:
        return min(0.20, len(meaningful_common) * 0.10)
    return 0.0


def calculate_similarity(query, candidate):
    """
    Calculate overall similarity score between query and candidate.
    Returns a score from 0.0 to 1.0.
    """
    base_score = similarity_ratio(query, candidate)
    bonus = contains_bonus(query, candidate) + word_match_bonus(query, candidate)
    
    return min(1.0, base_score + bonus)


# Place priority bonus (places are slightly prioritized over streets)
PLACE_PRIORITY_BONUS = 0.03  # 3% bonus for places


def find_combined_matches(query, place_names, street_names, top_n=5):
    """
    Find the best matching names from both places and streets.
    Places get a slight priority bonus.
    
    Args:
        query: The user's input string
        place_names: List of place names (from places_database.csv)
        street_names: List of street names (from map.json)
        top_n: Number of top matches to return
    
    Returns:
        List of tuples: [(name, similarity_score, type), ...] sorted by score descending
        where type is "place" or "street"
    """
    scores = []
    
    # Score places (with priority bonus)
    for name in place_names:
        score = calculate_similarity(query, name)
        # Add place priority bonus
        score_with_bonus = min(1.0, score + PLACE_PRIORITY_BONUS)
        scores.append((name, score_with_bonus, "place", score))  # Keep original score for display
    
    # Score streets (no bonus)
    for name in street_names:
        score = calculate_similarity(query, name)
        scores.append((name, score, "street", score))
    
    # Sort by score (with bonus) descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_n, but use original score for display
    return [(name, original_score, type_) for name, _, type_, original_score in scores[:top_n]]


def match_location(query, place_names, street_names, threshold=0.90):
    """
    Match user input to a place or street name with smart handling.
    
    Args:
        query: The user's input string
        place_names: List of place names
        street_names: List of street names
        threshold: Minimum similarity for automatic match (default 90%)
    
    Returns:
        tuple: (matched_name, matched_type, was_automatic, top_matches)
        - matched_name: The selected name (or None if user needs to choose)
        - matched_type: "place" or "street" (or None)
        - was_automatic: True if match was automatic (>=threshold)
        - top_matches: List of top 5 matches with scores and types
    """
    top_matches = find_combined_matches(query, place_names, street_names, top_n=5)
    
    if not top_matches:
        return None, None, False, []
    
    best_match, best_score, best_type = top_matches[0]
    
    # Check for exact match first (case-insensitive)
    query_lower = query.lower().strip()
    for name in place_names:
        if name.lower().strip() == query_lower:
            return name, "place", True, top_matches
    for name in street_names:
        if name.lower().strip() == query_lower:
            return name, "street", True, top_matches
    
    # If best match is above threshold, use it automatically
    if best_score >= threshold:
        return best_match, best_type, True, top_matches
    
    # Otherwise, return None and let the user choose
    return None, None, False, top_matches


def interactive_location_selection(query, place_names, street_names, point_type="point"):
    """
    Interactive selection of a location (place or street) with fuzzy matching.
    
    Args:
        query: The user's input string
        place_names: List of place names
        street_names: List of street names
        point_type: Description string ("starting point" or "ending point")
    
    Returns:
        tuple: (selected_name, selected_type) where type is "place" or "street"
    """
    matched_name, matched_type, was_automatic, top_matches = match_location(
        query, place_names, street_names
    )
    
    if was_automatic and matched_name:
        score = top_matches[0][1]
        print(f"  → Calculating {point_type} as: {matched_name}, {score*100:.0f}% accuracy ({matched_type})")
        return matched_name, matched_type
    
    # No automatic match - let user choose from top 5
    print(f"\n  No exact match found for '{query}'. Did you mean one of these?")
    for i, (name, score, type_) in enumerate(top_matches, 1):
        print(f"    {i}. {name}, {score*100:.0f}% accuracy ({type_})")
    print(f"    0. None of these (enter a different name)")
    
    while True:
        choice = input(f"\n  Select option (1-5, or 0): ").strip()
        
        if choice == "0":
            new_query = input("  Enter a different place or street name: ").strip()
            return interactive_location_selection(new_query, place_names, street_names, point_type)
        
        if choice in ["1", "2", "3", "4", "5"]:
            idx = int(choice) - 1
            if idx < len(top_matches):
                selected_name, _, selected_type = top_matches[idx]
                print(f"  → Calculating {point_type} as: {selected_name} ({selected_type})")
                return selected_name, selected_type
        
        print("  Invalid choice. Please enter 1-5, or 0.")
