import pytest
from unittest.mock import patch, MagicMock
from generate_plan import main

def test_generate_plan_main():
    with patch('argparse.ArgumentParser.parse_args') as mock_args:
        # Mock args
        mock_args.return_value.route_url = "http://example.com"
        mock_args.return_value.cp = 300
        mock_args.return_value.w_prime = 20000
        
        with patch('generate_plan.fetch_route') as mock_fetch:
            mock_fetch.return_value = MagicMock() # course_df
            
            with patch('generate_plan.optimize_pacing') as mock_opt:
                mock_opt.return_value = MagicMock() # plan_df
                
                with patch('generate_plan.export_tcx') as mock_export:
                    main()
                    mock_fetch.assert_called()
                    mock_opt.assert_called()
                    mock_export.assert_called()
