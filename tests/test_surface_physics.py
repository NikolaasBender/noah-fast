import unittest
import pandas as pd
import numpy as np
from planning.optimizer import optimize_pacing, get_speed

class TestSurfacePhysics(unittest.TestCase):
    def test_surface_friction_impact(self):
        """
        Verify that Gravel surface results in lower speed than Paved
        for the same gradient and power (conceptually).
        Since optimizer chooses power, we can check the physics function directly
        and then check the optimizer output behavior.
        """
        
        # 1. Direct Physics Check
        p_test = 200
        grad = 0
        speed_paved = get_speed(p_test, grad, crr=0.005) # Paved
        speed_gravel = get_speed(p_test, grad, crr=0.005 * 1.6) # Gravel
        
        print(f"Direct Physics Check (200W, 0%): Paved={speed_paved:.2f} m/s, Gravel={speed_gravel:.2f} m/s")
        self.assertTrue(speed_gravel < speed_paved, "Gravel should be slower")
        
    def test_optimizer_surface_aware(self):
        """
        Create a flat course with 1km Paved and 1km Gravel.
        Check if optimizer generates different speeds/powers.
        """
        # Create Dummy Course
        dist = np.arange(0, 2000, 10)
        df = pd.DataFrame({'distance': dist})
        df['elevation'] = 100 # Flat
        df['gradient'] = 0.0
        df['lat'] = 0
        df['lon'] = 0
        
        # Set Surface
        # 0-1000m: Paved
        # 1000-2000m: Gravel
        df['surface'] = 'Paved'
        df.loc[df['distance'] >= 1000, 'surface'] = 'Gravel'
        
        # Optimize
        cp = 300
        w_prime = 20000
        plan = optimize_pacing(df, cp, w_prime, rider_mass=80)
        
        # Analyze Results
        # Get average speed for first half vs second half
        paved_section = plan[plan['distance'] < 1000]
        gravel_section = plan[plan['distance'] >= 1000]
        
        avg_speed_paved = paved_section['speed_mps'].mean()
        avg_speed_gravel = gravel_section['speed_mps'].mean()
        
        print(f"Optimizer Results: Paved Speed={avg_speed_paved:.2f}, Gravel Speed={avg_speed_gravel:.2f}")
        
        self.assertTrue(avg_speed_gravel < avg_speed_paved, "Optimizer should predict lower speed on gravel")
        
        # Also check if it segmented correctly (should typically splits segments at surface boundary)
        # Note: resampled to 100m, so index 10 should be likely boundary
        self.assertGreater(len(plan['segment_id'].unique()), 1, "Should split into at least 2 segments")

if __name__ == '__main__':
    unittest.main()
