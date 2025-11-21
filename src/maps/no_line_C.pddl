(define (problem yes-line-x)
    (:domain multi_agent_lines)
    (:objects tank1 tank2 tank3 tank4 tank5 tank6 - tank well1 well2 well3 well4 well5 well6 well7 well8 well9 well10 well11 well12 well13 well14 - well agent1 agent2 agent3 agent4 agent5 agent6 agent7 agent8 - agent)
    (:init  
        (= (walls)
            (transpose (bit-mat
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 1 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 1 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 1 0 1 1)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 1 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 1 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 1 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 1 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 1 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0))))

        (= (xloc tank1) 1)
        (= (yloc tank1) 5)
        (= (xloc tank2) 1)
        (= (yloc tank2) 6)
        (= (xloc tank3) 1)
        (= (yloc tank3) 7)
        (= (xloc tank4) 1)
        (= (yloc tank4) 8)
        (= (xloc tank5) 1)
        (= (yloc tank5) 9)
        (= (xloc tank6) 1)
        (= (yloc tank6) 10)

        (= (xloc well1) 6)
        (= (yloc well1) 3)

        (= (xloc well2) 7)
        (= (yloc well2) 4)

        (= (xloc well3) 8)
        (= (yloc well3) 5)

        (= (xloc well4) 9)
        (= (yloc well4) 6)

        (= (xloc well5) 9)
        (= (yloc well5) 7)

        (= (xloc well6) 10)
        (= (yloc well6) 7)

        (= (xloc well7) 10)
        (= (yloc well7) 8)

        (= (xloc well8) 1)
        (= (yloc well8) 11)

        (= (xloc well9) 2)
        (= (yloc well9) 12)

        (= (xloc well10) 3)
        (= (yloc well10) 13)

        (= (xloc well11) 4)
        (= (yloc well11) 13)

        (= (xloc well12) 5)
        (= (yloc well12) 13)

        (= (xloc well13) 6)
        (= (yloc well13) 13)

        (= (xloc well14) 7)
        (= (yloc well14) 14)

        (= (xloc agent1) 12)
        (= (yloc agent1) 7)
        
        (= (xloc agent2) 12)
        (= (yloc agent2) 8)

        (= (xloc agent3) 12)
        (= (yloc agent3) 9)

        (= (xloc agent4) 12)
        (= (yloc agent4) 10)

        (= (xloc agent5) 12)
        (= (yloc agent5) 11)

        (= (xloc agent6) 12)
        (= (yloc agent6) 12)

        (= (xloc agent7) 12)
        (= (yloc agent7) 13)

        (= (xloc agent8) 12)
        (= (yloc agent8) 14)
    )
        
    (:goal 
        (and 
            (has-filled agent1) 
            (has-filled agent2) 
            (has-filled agent3) 
            (has-filled agent4) 
            (has-filled agent5) 
            (has-filled agent6) 
            (has-filled agent7) 
            (has-filled agent8)

            (= (xloc agent1) 14)
            (= (yloc agent1) 14)
            (= (xloc agent2) 13)
            (= (yloc agent2) 14)
            (= (xloc agent3) 12)
            (= (yloc agent3) 14)
            (= (xloc agent4) 11)
            (= (yloc agent4) 14)
            (= (xloc agent5) 10)
            (= (yloc agent5) 14)
            (= (xloc agent6) 9)
            (= (yloc agent6) 14)
            (= (xloc agent7) 13)
            (= (yloc agent7) 13)
            (= (xloc agent8) 13)
            (= (yloc agent8) 12)
        )
    )
)