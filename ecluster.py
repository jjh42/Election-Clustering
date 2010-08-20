# Do the election clustering a make some pretty figures.

from pylab import *;
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import rankdata
import sys
from scipy.spatial import distance


def convert_array(a, column):
    """Extract column from the array of lines and return as an individual array.
    Also get rid of any \" and other junk."""

    items = []
    for line in a:
        l = line[column]
        # Clean l to get rid of any "
        if type(l) == string_: # If a numpy string
            l = l.tostring()
            l = l.strip('"') # Get rid of " and any extra spaces
            l = l.strip()
        items.append(l)

    return array(items)


class Group():
    """A group, is a party or a set of independents running under the same ticket."""
    def __init__(self, ticket, name, surnames='', party=''):
        self.ticket=ticket
        self.name=name
        # The matching to the preference list is done by party or
        # for the independents by surname
        if party:
            self.party=party
            self.type = 'party'
        else:
            self.surnames = surnames
            self.type = 'independent'
            
    def match(self, surname, party):
        'Check if we match the description.'
        if self.type == 'independent':
            matches = filter(lambda x: x == surname, self.surnames)
            if matches:
                return True
        else:
            return party == self.party

        return False

def find_groups(groups_filename):
    """Using the electoral data find a list of the parties including associated
    independents and an appropriate name for each party."""
    
    group_data = loadtxt(groups_filename,
                     dtype={'names' : ('Event','State','Group','Surname','GivenName','Party'),
                            'formats' : ('S30', 'S30', 'S30', 'S30', 'S30', 'S100')},
                     delimiter=',', skiprows=1)

    group_tickets = convert_array(group_data, 2)
    group_surnames = convert_array(group_data,3)
    group_firstname = convert_array(group_data, 4)
    group_party = convert_array(group_data,5)

    groups = []

    for g in unique(group_tickets):
        # Ignore the 'UV' ticket - this are groups that don't have an above the line ticket
        if g == 'UG':
            continue

        # If this group has a party name
        if len(group_party[group_tickets == g][0]):
            gname = group_party[group_tickets == g][0]
            party = gname
            new_g = Group(ticket=g, name=gname, party=party)
        else:
            # This is a set of independents running so just choose something nice
            gname = 'Ind. ' + reduce(lambda x, y: x + ' & ' + y,
                                     group_surnames[group_tickets == g])

            new_g = Group(ticket=g, name=gname, surnames=group_surnames[group_tickets == g])

        groups.append(new_g)

    return groups

def load_au_preferencing_data(tickets_filename, groups_filename):
    """Load senate preferencing data provide by AEC"""

    data = loadtxt(tickets_filename,
                   dtype={'names' : ('StataAb','Ticket', 'TicketNo', 'Surname', 'GivenName',
                                            'Party', 'Preference'),
                                 'formats' : ('S5', 'S10', 'S5', 'S25', 'S25', 'S100', 'int16')},
                   delimiter=',',skiprows=1)
    # Convert data to individual arrays for easier manipulation
    ticket = convert_array(data, 1)
    ticketNo = array(convert_array(data,2), dtype='int8')
    surname = convert_array(data, 3)
    firstname = convert_array(data,4)
    party = convert_array(data, 5)
    preference = convert_array(data,6)

    groups = find_groups(groups_filename)
    
    return (groups, {'ticket' : ticket, 'ticketNo' : ticketNo, 'surname' : surname, 'firstname' : firstname,
                     'party' : party, 'preference' : preference})


def match_group(surname, party, groups):
    """Find the group which matches using either the party name
    or the surname. Return the index of the matching group"""
    o = filter(lambda g: g.match(surname, party), groups)
    assert(len(o) < 2)
    if o:
        return groups.index(o[0])
    else:
        return -1

def plot_clustering(D, groups):
    """Given the similarity matrix now calculate similarity."""

    # Make the matrix D into a valid similarity matrix
    D = (D + D.transpose()) / 2

    assert(distance.is_valid_dm(D))
    Z = linkage(D)

    # Generate a set of labels as well.
    labels = map(lambda g: g.name, groups)


    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=20)
    show()


def calc_distance_matrix(groups, preferences):
    """Turns preferences into a non-symmetric
    association matrix (not necessarily following a true metric).

    0 indicates total agreement, 1 indicates total disagreement.

    We group by ticket (party). We then find the mean preference given to other tickets and
    then rang the other tickets by the mean prefence. We ignore parties that don't have
    an above-the-line ticket (a few independents).

    i.e. Labor rates 0 Labor, 1 Greens, 2 Democrats ... 12 Family First.

    We use ordering because otherwise our results are affect by how many canidates a
    party has.

    Denote the ranking of each party j, given by party i as r_ij. The max(r_ij) will be the same for
    all i. When there are multiple tickets we average across the tickets.

    We then calculate a distance each party is from Labor (as ranked by Labor, the ranking
    the other party gave Labor is only important for calculating their ranking). This is why
    our matrix is not symmetric.

    d(t_i, t_j) = [ r_ij / max(r_ij) ]
    """

    D = zeros([len(groups), len(groups)])

    for g, ind in zip(groups, range(len(groups))):
        """Calculate the row of the distance matrix for this group."""
        mask = preferences['ticket'] == g.ticket
        this_sur = preferences['surname'][mask]
        this_parties = preferences['party'][mask]
        this_preference = preferences['preference'][mask]

        # We ignore the ticketNo columns because our algorithm with average across both tickets
        # for groups with two tickets

        # For each entry in this groups preferences we find the group it is associated with
        # and the preference they gave it and we record it.
        row_preferences = zeros(len(groups))
        row_count = zeros(len(groups))
        for sur,party,preference in zip(this_sur, this_parties, this_preference):
            gindex = match_group(sur, party, groups)
            if gindex >= 0:
                row_preferences[gindex] += preference
                row_count[gindex] += 1

        row_preferences = row_preferences/row_count
        # Now generate a row ranking based on these preferences
        row_ranking = rankdata(row_preferences) - 1
        row_metric = row_ranking/max(row_ranking)
        D[:][ind] = row_metric

    return D

def main():
    """Do the election cluster and print all figures."""
    filename = sys.argv[1]
    group_file = sys.argv[2]
    print('Loading from %s' % filename)

    (groups, preferences) = load_au_preferencing_data(filename, group_file)
    D = calc_distance_matrix(groups, preferences)
    plot_clustering(D, groups)

if __name__ == '__main__':
    main()
