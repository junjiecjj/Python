#pragma once

class CTrellis
{
public:
	CTrellis(void);
	~CTrellis(void);
public:
private:
	int m_num_state;
	int m_num_input;
	int m_num_output;

	int* m_in_label;

};
