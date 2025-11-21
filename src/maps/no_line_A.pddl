(define (problem yes-line-x)
    (:domain multi_agent_lines)
    (:objects tank1 tank2 tank3 tank4 tank5 tank6 - tank well1 well2 well3 well4 well5 well6 well7 well8 well9 well10 well11 well12 - well agent1 agent2 agent3 agent4 agent5 agent6 agent7 agent8 - agent)
    (:init  
        (= (walls)
            (transpose (bit-mat
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
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
        (= (yloc well1) 1)

        (= (xloc well2) 7)
        (= (yloc well2) 2)

        (= (xloc well3) 7)
        (= (yloc well3) 3)

        (= (xloc well4) 8)
        (= (yloc well4) 4)

        (= (xloc well5) 9)
        (= (yloc well5) 5)

        (= (xloc well6) 9)
        (= (yloc well6) 6)

        (= (xloc well7) 10)
        (= (yloc well7) 7)

        (= (xloc well8) 11)
        (= (yloc well8) 8)

        (= (xloc well9) 12)
        (= (yloc well9) 9)

        (= (xloc well10) 13)
        (= (yloc well10) 10)

        (= (xloc well11) 14)
        (= (yloc well11) 11)

        (= (xloc well12) 14)
        (= (yloc well12) 12)

        (= (xloc agent1) 11)
        (= (yloc agent1) 10)
        
        (= (xloc agent2) 10)
        (= (yloc agent2) 10)

        (= (xloc agent3) 9)
        (= (yloc agent3) 10)

        (= (xloc agent4) 8)
        (= (yloc agent4) 10)

        (= (xloc agent5) 7)
        (= (yloc agent5) 10)

        (= (xloc agent6) 6)
        (= (yloc agent6) 10)

        (= (xloc agent7) 5)
        (= (yloc agent7) 10)

        (= (xloc agent8) 4)
        (= (yloc agent8) 10)
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