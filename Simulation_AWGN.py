import numpy as np

########################################################################################################################
#                                                                                                                      #
#                                              Reliability Sequence                                                    #
#                                                                                                                      #
########################################################################################################################

Raw_sequence ="1 2 3 5 9 17 33 4 6 65 10 7 18 11 19 129 13 34 66 21 257 35 25 37 8 130 67 513 12 41 69 131 20 14 49 15 73 258 22 133 36 259 27 514 81 38 26 23 137 261 265 39 515 97 68 42 145 29 70 43 517 50 75 273 161 521 289 529 193 545 71 45 132 82 51 74 16 321 134 53 24 135 385 77 138 83 57 28 98 40 260 85 139 146 262 30 44 99 516 89 141 31 147 72 263 266 162 577 46 101 641 52 149 47 76 267 274 518 105 163 54 194 153 78 165 769 269 275 519 55 84 58 522 113 136 79 290 195 86 277 523 59 169 140 100 87 61 281 90 291 530 525 197 142 102 148 177 143 531 322 32 201 91 546 293 323 533 264 150 103 106 305 297 164 93 48 268 386 547 325 209 387 151 154 166 107 56 329 537 578 549 114 155 80 270 109 579 225 167 520 553 196 271 642 524 276 581 292 60 170 561 115 278 157 88 198 117 171 62 532 526 643 282 279 527 178 294 389 92 585 770 199 173 121 202 337 63 283 144 104 179 295 94 645 203 593 324 393 298 771 108 181 152 210 285 649 95 205 299 401 609 353 326 534 156 211 306 548 301 110 185 535 538 116 168 226 327 307 773 158 657 330 111 118 213 172 777 331 227 550 539 388 309 217 417 272 280 159 338 551 673 119 333 580 541 390 174 122 554 200 785 180 229 339 313 705 391 175 555 582 394 284 123 449 354 562 204 64 341 395 528 583 557 182 296 286 233 125 206 183 644 563 287 586 300 355 212 402 186 397 345 587 646 594 536 241 207 96 328 565 801 403 357 308 302 418 214 569 833 589 187 647 405 228 897 595 419 303 650 772 361 540 112 332 215 310 189 450 218 409 610 597 552 651 230 160 421 311 542 774 611 658 334 120 601 340 219 369 653 231 392 314 451 543 335 234 556 775 176 124 659 613 342 778 221 315 425 396 674 584 356 288 184 235 126 558 661 617 343 317 242 779 564 346 453 398 404 208 675 559 786 433 358 188 237 665 625 588 781 706 127 243 566 399 347 457 359 406 304 570 245 596 190 567 677 362 707 590 216 787 648 349 420 407 465 681 802 363 591 410 571 789 598 573 220 312 709 599 602 652 422 793 803 612 603 411 232 689 654 249 370 191 365 655 660 336 481 316 222 371 614 423 426 452 615 544 236 413 344 373 776 318 223 427 454 238 560 834 805 713 835 662 809 780 618 605 434 721 817 837 348 898 244 663 455 319 676 619 899 782 377 429 666 737 568 841 626 239 360 458 400 788 592 679 435 678 350 246 459 667 621 364 128 192 783 408 437 627 572 466 682 247 708 351 600 669 791 461 250 683 574 412 804 790 710 366 441 629 690 375 424 467 794 251 372 482 575 414 604 367 469 656 901 806 616 685 711 430 795 253 374 606 849 691 714 633 483 807 428 905 415 224 664 693 836 620 473 456 797 810 715 722 838 717 865 811 607 913 723 697 378 436 818 320 622 813 485 431 839 668 489 240 379 460 623 628 438 381 819 462 497 670 680 725 842 630 352 468 439 738 252 463 443 442 470 248 684 843 739 900 671 784 850 821 729 929 792 368 902 631 686 845 634 712 254 692 825 903 687 741 851 376 445 471 484 416 486 906 796 474 635 745 853 961 866 694 798 907 716 808 475 637 695 255 718 576 914 799 812 380 698 432 608 490 867 724 487 909 719 814 477 857 840 726 699 915 753 869 820 815 440 930 491 624 672 740 917 464 844 382 498 931 822 727 962 873 493 632 730 701 444 742 846 921 383 823 852 731 499 881 743 446 472 636 933 688 904 826 501 847 746 827 733 447 963 937 476 854 868 638 908 488 696 747 829 754 855 858 505 800 256 965 910 720 478 916 639 749 945 870 492 700 755 859 479 969 384 911 816 977 871 918 728 494 874 702 932 757 861 500 732 824 923 875 919 503 934 744 761 882 495 703 922 502 877 848 993 448 734 828 935 883 938 964 748 506 856 925 735 830 966 939 885 507 750 946 967 756 860 941 831 912 872 640 889 480 947 751 970 509 862 758 971 920 876 863 759 949 978 924 973 762 878 953 496 704 936 979 884 763 504 926 879 736 994 886 940 995 981 927 765 942 968 887 832 948 508 890 985 752 943 997 972 891 510 950 974 1001 893 951 864 760 1009 511 980 954 764 975 955 880 982 983 928 996 766 957 888 986 998 987 944 892 999 767 512 989 1002 952 1003 894 976 895 1010 956 1005 1011 958 984 959 988 1013 1000 1017 768 990 1004 991 1006 960 1012 1014 896 1007 1015 1018 1019 992 1021 1008 1016 1020 1022 1023 1024"

test = "1|2|3|5|9|17|33|4|6|7|10|11|13|18|19|21|25|34|35|37|41|49|8|12|14|15|20|22|23|26| 27|29|36|38|39|42|43|45|50|51|53| 57|16|24|28| 30|31|40|44|46|47|52|54|55|58|59|61|32|48|56|60|62|63|64"

Reliability_sequence = list(map(int, Raw_sequence.split()))
Test_sequence = list(map(int, test.split('|')))

# Extract the N-long reliability sequence from the 1024 reliability sequence
def extract_RS_N(long_RS, N):
    RS_N = []
    for s in long_RS:
        if (s <= N):
            RS_N.append(s)
    return RS_N

########################################################################################################################
#                                                                                                                      #
#                                          Encoder + AWGN/BPSK Transmission                                            #
#                                                                                                                      #
########################################################################################################################

# Put the message in u according the control sequence
def initialize_u(message, sequence, N, K):
    # Initialise u to all zeros -> the bits we don't touch are frozen
    u = np.zeros(N, dtype=int)
    # Place the message in the "best channel" bits
    b = 0
    for i in sequence[N-K:N]:
        u[i-1] = message[b]
        b += 1
    return u

def node_sum(left, right, l):
    result = np.zeros(l, dtype=int)
    for i in range(l):
        result[i] = np.mod(right[i] + left[i],2)
    return result

def encode(u, N):
    # Define the depth of the tree. We begin at the bottom
    d = int(np.log2(N))
    # nb is the  number of bit to combine at each step of the algorithm
    nb = 1
    while(d>0):

        for i in range(0,N,2*nb):
            left = u[i:i+nb]
            right = u[i+nb:i+2*nb]
            for j in range(nb):
                u[j+i] = node_sum(left, right, nb)[j]
        d -=1
        nb *=2
    return u

def transmitAWGN_BPSK(cword, sigma):
    mod_u = np.zeros(len(cword))
    noise = np.random.normal(0, sigma, len(cword))
    for i in range(len(cword)):
        mod_u[i] = 1-2*cword[i]
    return mod_u + noise

########################################################################################################################
#                                                                                                                      #
#                                                   Decoder                                                            #
#                                                                                                                      #
########################################################################################################################

def f(a,b):
    l = len(a)
    F_ab = np.zeros(l)
    for i in range(l):
        prod_tanh = np.tanh(a[i] / 2) * np.tanh(b[i] / 2)
        prod_tanh = np.clip(prod_tanh, -1 + 1e-10, 1 - 1e-10)  # Limit to avoid dividing by zero
        F_ab[i] = 2 * np.arctanh(prod_tanh)
        """F_ab[i] = 2*np.arctanh(np.tanh(a[i]/2)*np.tanh(b[i]/2))"""
    return F_ab

def g(a,b,c):
    l = len(a)
    G_ab = np.zeros(l)
    for i in range(l):
        G_ab[i] = (-1)**c[i] *(a[i]) + b[i]
    return G_ab

def u(ucapl, ucapr):
    l_plus_r = np.zeros(len(ucapl))
    for i in range(len(ucapl)):
        l_plus_r[i] = np.mod(ucapl[i] + ucapr[i], 2)
    return np.concatenate((l_plus_r, ucapr))

def insert_vector(put, receive, start):
    for i in range(len(put)):
        receive[i + start] = put[i]

def polar_decode(received, N, K, RS_N):
    max_depth = int(np.log2(N))

    # Storage for the beliefs, size: Depth lines, N columns
    Beliefs = np.zeros([max_depth + 1, N])

    # Storage for the data coming back
    ucap = np.zeros([max_depth + 1, N])

    # Storage for the nodes state: 0 = yet to be activated, 1 = finish L, 2 = finish R, 3 = finish U
    Nodes_state = np.zeros(2 ** (max_depth + 1) - 1)

    # Initialize the roots belief
    Beliefs[0] = received

    # Initialize the crossing of the tree from the root
    node = 0
    depth = 0

    # Starting the decoding while it is not finish
    done = 0
    nearly_done = 0
    while done == 0:
        # Check if we are at the leaf position
        if depth == max_depth:
            # Check if the leaf is a frozen bit
            if (node + 1) in RS_N[:N - K]:
                ucap[depth][node] = 0
                """print("node", node, depth, "is frozen")"""
            # The non-frozen case is the hard decision for ui
            else:
                """print("node", node, depth, "is not frozen")"""
                if Beliefs[depth][node] > 0:
                    ucap[depth][node] = 0
                else:
                    ucap[depth][node] = 1
            # Check if the decoding is complete
            if node == N - 1:
                nearly_done = 1
            # Going back to parent
            node = int(node / 2)
            depth -= 1
        else:
            # Position of the node in the node state vector
            node_position = int(2 ** depth - 1 + node)

            # Step L and go to left child
            if Nodes_state[node_position] == 0:
                """print("L and go to left child:")
                print("Node :", node, "---   Depth: ", depth)"""
                # Length of the incoming belief in bits
                nb = int(N / (2 ** (depth + 1)))

                # Extracting the incoming belief from Beliefs
                Incoming_belief = Beliefs[depth][2 * nb * node:2 * nb * node + 2 * nb]
                a = Incoming_belief[:nb]
                b = Incoming_belief[nb:]
                # Going to the left child
                node *= 2
                depth += 1
                # nb *= 0.5
                # Sent the belief to the child
                insert_vector(f(a, b), Beliefs[depth], node * nb)
                Nodes_state[node_position] = 1
            # Step R and go to right child
            elif Nodes_state[node_position] == 1:
                """print("R and go to right child:")
                print("Node :", node, "---   Depth: ", depth)"""
                # Length of the incoming belief in bits
                nb = int(N / (2 ** (depth + 1)))
                # Extracting the incoming belief from Beliefs
                Incoming_belief = Beliefs[depth][2 * nb * node:2 * nb * node + 2 * nb]
                a = Incoming_belief[:nb]
                b = Incoming_belief[nb:]
                # Get the info from the left child
                left_child = 2 * node
                left_child_depth = depth + 1
                # Get the incoming decision from the left child
                u_hat = ucap[left_child_depth][left_child * nb:left_child * nb + nb]
                # Going to the right child
                node = 2 * node + 1
                depth += 1
                # nb *= 0.5
                # Sent the belief to the child
                insert_vector(g(a, b, u_hat), Beliefs[depth], node * nb)
                Nodes_state[node_position] = 2
            # Step U and go to parent
            elif Nodes_state[node_position] == 2:
                # Length of the incoming belief in bits
                nb = int(N / (2 ** (depth + 1)))
                # Get the info from the left and right children
                left_child = 2 * node
                right_child = 2 * node + 1
                child_depth = depth + 1
                # Get the incoming decision from the left child
                u_hat_left = ucap[child_depth][left_child * nb:left_child * nb + nb]
                # Get the incoming decision from the right child
                u_hat_right = ucap[child_depth][right_child * nb:right_child * nb + nb]
                insert_vector(u(u_hat_left, u_hat_right), ucap[depth], node * 2 * nb)
                # Going back to parent
                node = int(node / 2)
                depth -= 1
                if nearly_done > 0:
                    if nearly_done == max_depth:
                        done = 1
                    nearly_done += 1
    return ucap

def convert_to_message(guess, sequence):
    non_Frozen = sequence[int(N/2):]
    message = np.zeros(int(N/2))
    b = 0
    for i in non_Frozen:
        message[b] = guess[i-1]
        b += 1
    return message.astype(int)

########################################################################################################################
#                                                                                                                      #
#                                                   Simulation                                                         #
#                                                                                                                      #
########################################################################################################################

def simulation(N, S, EbNo_dB_range):
    EBNO = []
    WER = []
    BER = []
    RS_N = extract_RS_N(Reliability_sequence, N)
    max_depth = int(np.log2(N))

    for EbNo_dB in EbNo_dB_range:
        rate = N / K
        EbNo = 10 ** (EbNo_dB / 10)
        sigma = np.sqrt(1 / (2 * rate * EbNo))

        EBNO.append(EbNo_dB)

        bit_error = 0
        word_error = 0
        total_bits = N * S
        total_words = S

        for i in range(S):
            print("EbNo_dB: ", EbNo_dB, "Avancement: ", int(i/S*100), "%")
            message = np.random.choice([1, 0], size=K)
            U = encode(initialize_u(message, RS_N, N, K), N)
            ucaps = polar_decode(list(transmitAWGN_BPSK(U, sigma)), N, K, RS_N)
            decoded = ucaps[0]
            message_decoded = ucaps[max_depth]
            guess = convert_to_message(message_decoded, RS_N)

            for b in range(len(decoded)):
                if decoded[b] != U[b]:
                    bit_error += 1

            if not np.array_equal(message, guess):
                word_error += 1

        BER.append(bit_error / total_bits)
        WER.append(word_error / total_words)

    return EBNO, WER, BER

########################################################################################################################
#                                                                                                                      #
#                                              Simulation Parameters                                                   #
#                                                                                                                      #
########################################################################################################################

N = 256
K = 128
EbNo_dB_range = [0.0001, 0.0005, 0.001, 0.005,0.01, 0.05, 0.1, 0.5, 1, 5]
S = 1000000

print(simulation(N, S, EbNo_dB_range))