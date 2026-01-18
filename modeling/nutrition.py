def calculate_metabolic_cost(power_watts, efficiency=0.23):
    """
    Calculates metabolic cost in kcal/hour.
    
    Args:
        power_watts: Average power.
        efficiency: Gross Metabolic Efficiency (20-25%). 23% is standard for trained cyclists.
        
    Returns:
        kcal_per_hour
    """
    # 1 Watt = 1 Joule/second
    # 1 kcal = 4184 Joules
    
    joules_per_hour = power_watts * 3600
    kcal_mechanical = joules_per_hour / 4184
    
    kcal_metabolic = kcal_mechanical / efficiency
    return kcal_metabolic

def carb_needs_per_hour(power_watts, weight_kg=75):
    """
    Estimates carbohydrate intake needs based on intensity.
    
    Rule of thumb:
    - < 50% FTP (Zone 1/2): 30g/hr (optional)
    - 60-75% FTP (Tempo): 60g/hr
    - > 80% FTP (Threshold/Race): 70-90g/hr (up to 120g/hr for pros)
    
    For a V1, we can use a linear mapping or zones. 
    Let's assume a generic FTP if unknown, or pass it in.
    But effectively, higher watts = higher carbs.
    
    Simple model: 
    - 1g carb per watt above 150w? No that's too simple.
    - Let's map kcal burn to CHO contribution.
    """
    kcal_burn = calculate_metabolic_cost(power_watts)
    
    # At high intensity, ~100% of fuel is carbs.
    # At low intensity, ~50% is fat.
    # Let's simple heuristic: Aim to replace 50% of caloric burn with carbs?
    # 1g carb = 4 kcal.
    
    # 200W -> ~750kcal/hr. 50% = 375kcal. 375/4 = ~94g carbs. Little high but okay for racing.
    # 150W -> ~560kcal/hr. 50% = 280kcal. 280/4 = 70g carbs.
    
    target_carbs = (kcal_burn * 0.5) / 4
    
    # Cap at absorption limits unless "gut training" logic
    return min(target_carbs, 100)
