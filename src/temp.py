        _x, _y = np.meshgrid(ALPHA, ENERGIES, indexing="ij")
        curr_possible_coordinates = np.array(list(itertools.product(curr_alpha, curr_energies)))
        
        for T in range(len(curr_ect_JD)):
            
            curr_possible_PSD = curr_PSD[T, :, :].flatten()
            not_nan = np.isfinite(curr_possible_PSD)
            valid_PSD = curr_possible_PSD[not_nan]
            valid_coordinates = curr_possible_coordinates[not_nan]
            interpolated_PSD = scipy.interpolate.griddata(points = valid_coordinates, values = valid_PSD, xi = (_x, _y), fill_value=np.NaN, method="linear")
            PSD = np.concatenate((PSD, interpolated_PSD), axis=0)
            
        EPOCH = np.concatenate((EPOCH, curr_ect_epoch), axis = 0)
        JD = np.concatenate((JD, curr_ect_JD), axis = 0)
        
        
        curr_B = np.expand_dims(np.expand_dims(np.interp(curr_ect_JD, curr_emfisis_JD, curr_emfisis_B, left=np.NAN, right=np.NaN), 1), 1)
        #curr_MU = np.expand_dims(np.outer(np.sin(ALPHA) ** 2, ((ENERGIES ** 2) + 2 * M_e * ENERGIES)), 0) / (2 * M_e * curr_B)
        
        
            ENERGIES = np.array([1.4942348e-05, 1.6766295e-05, 1.8800700e-05, 2.1115711e-05, 2.3641176e-05,
                        2.6517402e-05, 2.9744388e-05, 3.3392287e-05, 3.7461097e-05, 4.2020965e-05,
                        4.7142050e-05, 5.2824351e-05, 5.9278322e-05, 6.6503962e-05, 7.4571435e-05,
                        8.3621024e-05, 9.3793031e-05, 1.0522779e-04, 1.1799542e-04, 1.3230641e-04,
                        1.4844134e-04, 1.6647036e-04, 1.8667411e-04, 2.0940331e-04, 2.3486841e-04,
                        2.6342022e-04, 2.9540950e-04, 3.3132723e-04, 3.7166453e-04, 4.1684232e-04,
                        4.6749198e-04, 5.2431499e-04, 5.8808312e-04, 6.5956777e-04, 7.3975133e-04,
                        8.2975620e-04, 9.3063462e-04, 1.0437194e-03, 1.1706240e-03, 1.3129623e-03,
                        1.4726277e-03, 1.6516555e-03, 1.8524298e-03, 2.0776878e-03, 2.3303046e-03,
                        2.6135778e-03, 2.9313657e-03, 3.2877370e-03, 3.6874623e-03, 4.1358024e-03,
                        4.6386514e-03, 5.2026021e-03, 5.8351620e-03, 6.5446068e-03, 7.3403395e-03,
                        8.2327416e-03, 9.2337383e-03, 1.0356379e-02, 1.1615465e-02, 1.3027690e-02,
                        1.4611581e-02, 1.6388107e-02, 1.8380560e-02, 2.0615315e-02, 2.3121702e-02,
                        2.5932897e-02, 2.9085804e-02, 3.2622088e-02, 3.6588330e-02, 3.8236111e-02,
                        4.1036736e-02, 4.6026003e-02, 5.1621880e-02, 5.8137767e-02, 8.1608824e-02,
                        1.1054863e-01, 1.4558503e-01, 1.8313929e-01, 2.2061732e-01, 3.1548694e-01,
                        3.4753418e-01, 4.4909909e-01, 5.8040935e-01, 7.3232502e-01, 8.6909151e-01,
                        1.0309292e+00, 1.1662041e+00, 1.6496121e+00, 1.7679559e+00, 2.0982158e+00,
                        2.3059926e+00, 2.5885820e+00, 2.6795702e+00, 3.3927250e+00, 4.1784725e+00,
                        5.1864066e+00, 6.2901530e+00, 7.6955447e+00, 9.8988981e+00]) #This is the energies we will interpolate to in MeV
            
            
                valid_energy_channels = (0 < curr_ect_fedu_energy) & ((curr_ect_fedu_energy / 1000) < 10) & (curr_ect_fedu_energy_delta_plus > 0) & (curr_ect_fedu_energy_delta_minus > 0)
