import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
from zipfile import ZipFile
import numpy as np
from one.api import ONE
from scipy.stats import bernoulli
from tqdm import tqdm
import wget

# --------------------
# Constants and Config
# --------------------
IBL_DATA_PATH = "ibl_data/"
IBL_DATA_URL = 'https://ndownloader.figshare.com/files/21623715'
IBL_ZIP_FILENAME = "ibl-behavior-data-Dec2019.zip"
REQUIRED_NUM_SESSIONS = 15
DATA_BY_ANIMAL_FILE = "ibl_data_by_animal_dict.json"

# --------------------
# Utility Functions
# --------------------

def ensure_data_downloaded():
    """Ensure IBL data is downloaded and extracted."""
    if not os.path.exists(IBL_DATA_PATH):
        os.makedirs(IBL_DATA_PATH)
        print(f"Downloading IBL data...")
        wget.download(IBL_DATA_URL, IBL_DATA_PATH)
        with ZipFile(os.path.join(IBL_DATA_PATH, IBL_ZIP_FILENAME), 'r') as zipObj:
            zipObj.extractall(IBL_DATA_PATH)

def date_chunks(start_date, end_date, chunk_days=30):
    """Yield (start, end) date strings in chunks."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while start <= end:
        chunk_end = start + timedelta(days=chunk_days)
        yield (start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        start = chunk_end + timedelta(days=1)

# ------------------------------------------------
# Original Data Processing Functions from GLM-HMM (https://github.com/zashwood/glm-hmm)
# ------------------------------------------------

def remap_choice_vals(choice):
    """Remap raw choice vector: CW=1->0, CCW=-1->1, viol=0->-1."""
    choice_mapping = {1: 0, -1: 1, 0: -1}
    return [choice_mapping[old_choice] for old_choice in choice]

def get_raw_data(one, eid):
    """Load raw data for a session."""
    info = one.get_details(eid)
    animal = info['subject']
    session_id = info['start_time'][:10] + '-' + str(info['number'])
    choice = one.load_dataset(eid, '_ibl_trials.choice')
    stim_left = one.load_dataset(eid, '_ibl_trials.contrastLeft')
    stim_right = one.load_dataset(eid, '_ibl_trials.contrastRight')
    rewarded = one.load_dataset(eid, '_ibl_trials.feedbackType')
    bias_probs = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
    return animal, session_id, stim_left, stim_right, rewarded, choice, bias_probs

def create_stim_vector(stim_left, stim_right):
    """Create signed contrast vector (stim_right - stim_left), replacing NaNs with 0."""
    stim_left = np.nan_to_num(stim_left, nan=0)
    stim_right = np.nan_to_num(stim_right, nan=0)
    return stim_right - stim_left

def create_previous_choice_vector(choice):
    """Create previous choice vector, handling violations (-1)."""
    previous_choice = np.hstack([np.array(choice[0]), choice])[:-1]
    locs_to_update = np.where(previous_choice == -1)[0]
    locs_with_choice = np.where(previous_choice != -1)[0]
    loc_first_choice = locs_with_choice[0]
    locs_mapping = np.zeros((len(locs_to_update) - loc_first_choice, 2), dtype='int')
    for i, loc in enumerate(locs_to_update):
        if loc < loc_first_choice:
            previous_choice[loc] = bernoulli.rvs(0.5, 1) - 1
        else:
            potential_matches = locs_with_choice[np.where(locs_with_choice < loc)]
            absolute_val_diffs = np.abs(loc - potential_matches)
            absolute_val_diffs_ind = absolute_val_diffs.argmin()
            nearest_loc = potential_matches[absolute_val_diffs_ind]
            locs_mapping[i - loc_first_choice, 0] = int(loc)
            locs_mapping[i - loc_first_choice, 1] = int(nearest_loc)
            previous_choice[loc] = previous_choice[nearest_loc]
    assert len(np.unique(previous_choice)) <= 2, f"previous choice should be in {{0, 1}}; {np.unique(previous_choice)}"
    return previous_choice, locs_mapping

def create_wsls_covariate(previous_choice, success, locs_mapping):
    """Create win-stay-lose-switch covariate."""
    remapped_previous_choice = 2 * previous_choice - 1
    previous_reward = np.hstack([np.array(success[0]), success])[:-1]
    for i, loc in enumerate(locs_mapping[:, 0]):
        nearest_loc = locs_mapping[i, 1]
        previous_reward[loc] = previous_reward[nearest_loc]
    wsls = previous_reward * remapped_previous_choice
    assert len(np.unique(wsls)) == 2, "wsls should be in {-1, 1}"
    return wsls

def create_design_input_data(choice, stim_left, stim_right, rewarded):
    """Create design matrix input data."""
    stim = create_stim_vector(stim_left, stim_right)
    choice = remap_choice_vals(choice)
    previous_choice, locs_mapping = create_previous_choice_vector(choice)
    wsls = create_wsls_covariate(previous_choice, rewarded, locs_mapping)
    return stim, previous_choice, wsls

def get_all_unnormalized_data_this_session(one, eid):
    """Get all unnormalized data for a session, subset to 50-50 trials."""
    animal, session_id, stim_left, stim_right, rewarded, choice, bias_probs = get_raw_data(one, eid)
    trials_to_study = np.where(bias_probs == 0.5)[0]
    num_viols_50 = len(np.where(choice[trials_to_study] == 0)[0])
    if num_viols_50 < 10:
        input_data = create_design_input_data(choice[trials_to_study], stim_left[trials_to_study], stim_right[trials_to_study], rewarded[trials_to_study])
        y = remap_choice_vals(choice[trials_to_study])
        session = session_id
        rewarded = rewarded[trials_to_study]
    else:
        print(f"Skipping session {session_id} because it has {num_viols_50} violations")
        return None
    return animal, input_data, y, session, num_viols_50, rewarded

# --------------------
# Main Processing Logic
# --------------------

def main():
    ensure_data_downloaded()
    os.chdir(IBL_DATA_PATH)
    one = ONE()
    print(f"\nProcessing IBL data...")

    # Chunking by date to avoid API rate limiting
    all_eids = []
    for start, end in date_chunks("2018-01-01", "2021-12-31"):
        eids_chunk = one.search(date_range=[start, end], datasets=['_ibl_trials.choice.npy'])
        all_eids.extend(eids_chunk)

    animal_eid_dict = defaultdict(list)
    for eid in tqdm(all_eids):
        info = one.get_details(eid)
        bias_probs = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
        trained = np.all(np.isin([0.2, 0.5, 0.8], np.unique(bias_probs)))
        if trained:
            animal = info['subject']
            animal_eid_dict[animal].append(str(eid))

    animal_list = list(animal_eid_dict.keys())
    # Filter animals with enough sessions
    animal_list = [animal for animal in animal_list if len(animal_eid_dict[animal]) >= REQUIRED_NUM_SESSIONS]

    final_animal_data_dict = defaultdict(list)
    for z, animal in enumerate(animal_list):
        print(f"Processing animal {animal} ({z+1}/{len(animal_list)})")
        for eid in tqdm(animal_eid_dict[animal]):
            result = get_all_unnormalized_data_this_session(one, eid)
            if result is not None:
                animal_id, input_data, y, session, num_viols_50, rewarded = result
                stim, previous_choice, wsls = input_data
                final_animal_data_dict[animal].append({
                    'stim': stim.tolist(),
                    'previous_choice': previous_choice.tolist(),
                    'session': session,
                    'rewarded': rewarded.tolist()
                })

    # Write out animal data
    with open(os.path.join(IBL_DATA_PATH, DATA_BY_ANIMAL_FILE), "w") as f:
        json.dump(final_animal_data_dict, f)
    print(f"IBL data is ready!")

if __name__ == "__main__":
    main()
