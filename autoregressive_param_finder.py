# A script used to greedily find the best parameters for an autoregressive model
#
# The best parameters are the ones which:
#   - produce very different responses among each other
#   - produce responses which differ from other ar filters by > 2000

import argparse
import torch
from tqdm import tqdm
from autoregressive import ARProcessPerturb3Channel, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total', type=int, default=10, help='Number of AR processes to find')
    parser.add_argument('--num_samples', type=int, default=1000000, help='number of AR coefficients to sample')
    parser.add_argument('--num_gen_trials', type=int, default=3, help='number of patterns to generate using an AR process')
    parser.add_argument('--required_nm_response', type=int, default=10, help='lowest required response among non-matching filters')
    parser.add_argument('--gen_norm_upper_bound', type=int, default=50, help='upper bound on l_inf norm of generated patterns')
    parser.add_argument('--disable_tqdm', action='store_true', help='disable tqdm progress bar')
    args = parser.parse_args()

    num_samples = args.num_samples        # number of AR processes to sample
    num_gen_trials = args.num_gen_trials  # number of patterns to generate using an AR process

    REQUIRED_NM_RESPONSE = args.required_nm_response
    GEN_NORM_UPPER_BOUND = args.gen_norm_upper_bound
    TOTAL_AR = args.total

    ar_processes = []
    for i in tqdm(range(num_samples), disable=args.disable_tqdm):
        # create 3 channel AR process
        ar_p = ARProcessPerturb3Channel()

        # sample a few starting signals to ensure that signals don't diverge
        gen_norms = []
        for j in range(num_gen_trials):
            gen, gen_norm = ar_p.generate(eps=1.0, p=2, size=(36,36), crop=4)
            gen_norms.append(gen_norm)
        gen_norms = torch.stack(gen_norms)

        # if any generated signal diverges or is nan, regenerate AR process
        if torch.isnan(gen_norm) or (gen_norms > GEN_NORM_UPPER_BOUND).any():
            continue

        # ensure filters from other AR processes have high response
        # as measured by the lowest response from a non-matching filter
        # if not, regenerate AR process
        if len(ar_processes) > 0:
            responses = []
            for locked_ar in ar_processes:
                f = locked_ar.get_filter()
                resp = response(f, gen)
                responses.append(resp)
            lowest_response = min(responses)
            print('Lowest response', lowest_response)
            if lowest_response < REQUIRED_NM_RESPONSE:
                continue

            print(f'Found AR process with lowest response {lowest_response}')

        print('Adding process:')
        print(ar_p)
        ar_processes.append(ar_p)

        # if we've found enough AR processes, stop
        if len(ar_processes) == TOTAL_AR:
            break
    
    print(f'Completed with {len(ar_processes)} found!')
    coefficients_list = []
    for a in ar_processes:
        print(a)
        coefficients_list.append(a.b)

    # save coefficients as a list of torch tensors
    save_filename = f'params-classses-{TOTAL_AR}-mr-{REQUIRED_NM_RESPONSE}.pt'
    torch.save(coefficients_list, save_filename)
    print(f'Saved to {save_filename}!')


if __name__ == '__main__':
    main()