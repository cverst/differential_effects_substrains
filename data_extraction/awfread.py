import numpy as np
from datetime import datetime, date


def awf_read(path: str) -> dict:
    """TDT .awf file reader.

    Parameters
    ----------
    path : string
        String containing the path to the .awf file to be imported.

    Returns
    -------
    data : dictionary
        Dictionary containing all data from specified .awf file.
    """

    # Initialize parameters
    is_rz = False

    rec_head = dict()
    groups = []
    data = dict()

    with open(path, "rb") as fid:

        # Read RecHead data
        rec_head["nens"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        rec_head["ymax"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        rec_head["ymin"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        rec_head["autoscale"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        rec_head["size"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        rec_head["gridsize"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        rec_head["showgrid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        rec_head["showcur"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        rec_head["text_marg_left"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        rec_head["text_marg_top"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        rec_head["text_marg_right"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        rec_head["text_marg_bottom"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        b_first_pass = True

        for x in range(0, 30):

            # Create dict for looping
            loop_groups = dict()

            loop_groups["recn"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["grpid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            # Read temporary timestamp
            if b_first_pass:
                ttt = np.fromfile(fid, dtype=np.int64, count=1)
                fid.seek(-8, 1)
                # Make sure timestamps make sense.
                if (
                    datetime.now().toordinal()
                    - (ttt / 86400 + date.toordinal(date(1970, 1, 1)))
                    > 0
                ):
                    is_rz = True
                    data["file_time"] = datetime.fromtimestamp(ttt).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    data["file_type"] = "BioSigRZ"
                else:
                    ttt = np.fromfile(fid, dtype=np.uint32, count=1)
                    data["file_time"] = datetime.fromtimestamp(ttt).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    fid.seek(-4, 1)
                    data["file_type"] = "BioSigRP"
                b_first_pass = False

            if is_rz:
                loop_groups["grp_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups["grp_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            loop_groups["newgrp"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["sgi"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            # TODO: check why dtype int16 (from TDT specifications) does not work in next two lines
            loop_groups["chan"] = np.fromfile(fid, dtype=np.int8, count=1)[0]
            loop_groups["rtype"] = np.fromfile(fid, dtype=np.int8, count=1)[0]

            loop_groups["npts"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["osdel"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["dur"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["srate"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

            loop_groups["arthresh"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["gain"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["accouple"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            loop_groups["navgs"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["narts"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            if is_rz:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            tmp = np.zeros(10)
            for i in range(0, 10):
                tmp[i] = np.fromfile(fid, dtype=np.float32, count=1)
            loop_groups["vars"] = tmp

            cursors = []
            for i in range(0, 10):
                loop_cursors = dict()
                loop_cursors["tmar"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
                loop_cursors["val"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
                tmp_str = fid.read(20).decode("utf-8").split("\0")
                loop_cursors["desc"] = [
                    x for x in tmp_str if x and np.size(tmp_str) > 1
                ]
                loop_cursors["xpos"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors["ypos"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors["hide"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors["lock"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                cursors.append(loop_cursors)
            loop_groups["cursors"] = cursors

            # Open the group
            loop_groups["grpn"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["frecn"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["nrecs"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            tmp_str = fid.read(16).decode("utf-8").split("\0")
            loop_groups["id"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(16).decode("utf-8").split("\0")
            loop_groups["ref1"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(16).decode("utf-8").split("\0")
            loop_groups["ref2"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(50).decode("utf-8").split("\0")
            loop_groups["memo"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            if is_rz:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            tmp_str = fid.read(100).decode("utf-8").split("\0")
            loop_groups["sgfname1"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(100).decode("utf-8").split("\0")
            loop_groups["sgfname2"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name1"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name2"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name3"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name4"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name5"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name6"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name7"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name8"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name9"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["var_name10"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]

            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit1"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit2"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit3"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit4"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit5"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit6"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit7"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit8"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit9"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["var_unit10"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]

            loop_groups["samp_per_us"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

            loop_groups["cc_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            loop_groups["version"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["postproc"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            tmp_str = fid.read(92).decode("utf-8").split("\0")
            loop_groups["dump"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            loop_groups["bid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["comp"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["x"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["y"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            loop_groups["trace_cm"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["token_cm"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            loop_groups["col"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            loop_groups["curcol"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            blurb = []
            for i in range(0, 5):
                loop_blurb = dict()
                loop_blurb["type"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["incid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["hide"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["x"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["y"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["manplace"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                tmp_str = fid.read(12).decode("utf-8").split("\0")
                loop_blurb["txt"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
                blurb.append(loop_blurb)
            loop_groups["blurb"] = blurb
            loop_groups["ymax"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["ymin"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            tmp_str = fid.read(100).decode("utf-8").split("\0")
            loop_groups["equ"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            groups.append(loop_groups)

        for x in range(0, 30):
            if groups[x]["bid"] > 0 and groups[x]["npts"] > 0:
                npts = groups[x]["npts"]
                groups[x]["wave"] = np.fromfile(fid, dtype=np.float32, count=npts)
            else:
                groups[x]["wave"] = []

    data["rec_head"] = rec_head
    data["groups"] = groups

    return data
