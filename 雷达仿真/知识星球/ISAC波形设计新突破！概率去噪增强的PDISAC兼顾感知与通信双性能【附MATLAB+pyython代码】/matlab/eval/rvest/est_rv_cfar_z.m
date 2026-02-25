function [hat_r_tars, hat_v_tars] = est_rv_cfar_z(rdr, Z_PRBS_waveform, cfar2D, num_guard, num_refer, vrelmax, Rmax, num_of_targets)
    release(rdr);
    release(cfar2D);
    [resp, rng_grid, dop_grid] = rdr(Z_PRBS_waveform);
    [det_ranges, det_velocities, detected_powers] = applyCFAR2D(resp, rng_grid, dop_grid, cfar2D, num_guard(1), num_refer(1), vrelmax, Rmax);
    coords_targets = find_targets([det_ranges, det_velocities, detected_powers], 5);
    new_coords_targets = kmeans_target_detection(coords_targets, num_of_targets);
    hat_r_tars = new_coords_targets(:,1);
    hat_v_tars = new_coords_targets(:,2);
end
