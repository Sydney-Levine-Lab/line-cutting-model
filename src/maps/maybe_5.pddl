(define (problem yes-line-x)
    (:domain multi_agent_lines)
    (:objects tank1 tank2 tank3 tank4 tank5 tank6 - tank well1 well2 well3 well4 well5 well6 - well agent1 agent2 agent3 agent4 agent5 agent6 agent7 agent8 - agent)
    (:init  
        (= (walls)
            (transpose (bit-mat
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 1 1 1 0 0)
                (bit-vec 0 0 0 0 0 0 0 1 1 1 1 1 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                (bit-vec 0 0 0 0 0 0 0 1 1 1 1 1 0 0)
                (bit-vec 0 0 0 0 0 0 0 0 0 1 1 1 0 0)
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

        (= (xloc well1) 14)
        (= (yloc well1) 8)

        (= (xloc well2) 13)
        (= (yloc well2) 8)

        (= (xloc well3) 12)
        (= (yloc well3) 8)

        (= (xloc well4) 11)
        (= (yloc well4) 8)

        (= (xloc well5) 10)
        (= (yloc well5) 8)

        (= (xloc well6) 9)
        (= (yloc well6) 8)

        (= (xloc agent1) 6)
        (= (yloc agent1) 4)
        
        (= (xloc agent2) 7)
        (= (yloc agent2) 4)

        (= (xloc agent3) 8)
        (= (yloc agent3) 4)

        (= (xloc agent4) 9)
        (= (yloc agent4) 4)

        (= (xloc agent5) 10)
        (= (yloc agent5) 4)

        (= (xloc agent6) 11)
        (= (yloc agent6) 4)

        (= (xloc agent7) 12)
        (= (yloc agent7) 4)

        (= (xloc agent8) 13)
        (= (yloc agent8) 4)
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